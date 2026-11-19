// Immutable sets retain schedule storage until frame completion. Four compute
// stages publish indirect coverage bins; two raster epochs preserve depth ties.
NaniteHardwareWork HardwarePlan;
struct HardwareResources
{
    Impl& Owner; RawBuffer Uniform; VkDescriptorPool Pool=0;
    std::shared_ptr<NaniteFrameBuffer> Ranges,Arguments,Queue,Owners;
    std::array<VkDescriptorSet,2> ComputeSets{};
    std::vector<VkDescriptorSet> GraphicsSets;
    explicit HardwareResources(Impl& owner):Owner(owner) {}
    ~HardwareResources() { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); }
};
std::shared_ptr<HardwareResources> HardwareWork;
std::array<VkDescriptorSetLayout,2> HardwareSets{};
std::array<VkPipelineLayout,2> HardwareLayouts{};
std::array<VkPipeline,4> HardwareKernels{};
std::array<VkShaderModule,5> HardwareModules{};
std::unordered_map<unsigned,VkPipeline> HardwareGraphics;
VkRenderPass HardwareLoadPass=0;
void(EVANS_VK_CALL* HardwareDrawIndirect)(VkCommandBuffer,VkBuffer,VkDeviceSize,std::uint32_t,std::uint32_t)=nullptr;
void DestroyNaniteHardware()
{
    HardwarePlan={}; HardwareWork.reset();
    for (auto pipeline:HardwareKernels) if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); HardwareKernels={};
    for (auto& entry:HardwareGraphics) if (entry.second) DestroyPipelineFunction(Device,entry.second,nullptr); HardwareGraphics.clear();
    for (auto module:HardwareModules) if (module) DestroyShaderModuleFunction(Device,module,nullptr); HardwareModules={};
    for (auto layout:HardwareLayouts) if (layout) DestroyPipelineLayoutFunction(Device,layout,nullptr); HardwareLayouts={};
    for (auto set:HardwareSets) if (set) DestroyDescriptorSetLayoutFunction(Device,set,nullptr); HardwareSets={};
    if (HardwareLoadPass) DestroyRenderPassFunction(Device,HardwareLoadPass,nullptr); HardwareLoadPass=0;
}
VkPipeline NaniteHardwarePipeline(unsigned phase,bool cull,bool mirrored,bool mesh=false)
{
    const unsigned key=(phase<<2)|unsigned(cull)|(unsigned(mirrored)<<1)|(unsigned(mesh)<<4);
    const auto found=HardwareGraphics.find(key); if (found!=HardwareGraphics.end()) return found->second;
    VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,HardwareModules[phase==2?1:0],"main",nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,HardwareModules[2+phase],"main",nullptr}};
    if (mesh) { stages[0].stage=VK_SHADER_STAGE_MESH_BIT_EXT; stages[0].module=MeshModules[1]; stages[1].module=MeshFragmentModules[phase+1]; }
    VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; raster.polygonMode=VK_POLYGON_MODE_FILL;
    raster.cullMode=cull && phase!=2?CullBackFaces:VK_CULL_MODE_NONE; raster.frontFace=mirrored?VK_FRONT_FACE_CLOCKWISE:VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples=1;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}; depth.depthTestEnable=VK_TRUE_VALUE;
    depth.depthWriteEnable=phase==0?VK_TRUE_VALUE:VK_FALSE_VALUE; depth.depthCompareOp=phase==0?VK_COMPARE_OP_LESS:phase==1?VK_COMPARE_OP_EQUAL:VK_COMPARE_OP_ALWAYS; depth.maxDepthBounds=1;
    VkPipelineColorBlendAttachmentState colors[2]{};
    if (phase==2) for (auto& color:colors) color.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount=2; blend.pAttachments=colors;
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states;
    VkGraphicsPipelineCreateInfo desc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; desc.stageCount=2; desc.pStages=stages;
    desc.pVertexInputState=mesh?nullptr:&vertices; desc.pInputAssemblyState=mesh?nullptr:&assembly; desc.pViewportState=&viewport; desc.pRasterizationState=&raster;
    desc.pMultisampleState=&samples; desc.pDepthStencilState=&depth; desc.pColorBlendState=&blend; desc.pDynamicState=&dynamic;
    desc.layout=HardwareLayouts[1]; desc.renderPass=VisibilityRenderPass;
    VkPipeline pipeline=0; if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0,1,&desc,nullptr,&pipeline),"Create Nanite hardware graphics pipeline")) return 0;
    HardwareGraphics.emplace(key,pipeline); return pipeline;
}
bool PrepareNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteHardwareWork(frame,CreateInfo.Width,CreateInfo.Height,HardwarePlan,LastError)) return false;
    HardwareWork.reset(); if (!HardwarePlan.Active()) return true;
    const auto& c=HardwarePlan.Constants;
    const unsigned items=std::max(c.Counts.X,std::max(c.Counts.W,HardwarePlan.ArgumentWords));
    const unsigned rows=(items+NaniteHardwareThreads*NaniteHardwareColumns-1)/(NaniteHardwareThreads*NaniteHardwareColumns);
    if (!FragmentStoresAndAtomicsSupported || DeviceLimits.maxPerStageDescriptorStorageBuffers<4 || DeviceLimits.maxDescriptorSetStorageBuffers<4 ||
        NaniteHardwareColumns>DeviceLimits.maxComputeWorkGroupCount[0] || rows>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Vulkan hardware scheduling requires fragment atomics and four storage buffers");
    if (!LoadDevice(HardwareDrawIndirect,"vkCmdDrawIndirect")) return false;
    auto work=std::make_shared<HardwareResources>(*this);
    if (!CreateNaniteFrameBuffer(unsigned(HardwarePlan.Ranges.size()*4),HardwarePlan.Ranges.data(),false,work->Ranges) ||
        !CreateNaniteFrameBuffer(HardwarePlan.ArgumentWords,nullptr,true,work->Arguments,true) ||
        !CreateNaniteFrameBuffer(HardwarePlan.QueueWords,nullptr,true,work->Queue) ||
        !CreateNaniteFrameBuffer(c.Counts.W,nullptr,true,work->Owners)) return false;
    for (unsigned stage=0;stage<2;++stage)
    {
        if (!HardwareSets[stage])
        {
            VkDescriptorSetLayoutBinding bindings[8]{}; const unsigned count=stage?(MeshStagesEnabled?8:7):5;
            for (unsigned i=0;i<count;++i) bindings[i]={i,i==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(i<5 || i==7)?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,
                stage?VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|(MeshStagesEnabled?VK_SHADER_STAGE_MESH_BIT_EXT:0):VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
            VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=count; desc.pBindings=bindings;
            if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&HardwareSets[stage]),"Create hardware schedule descriptors")) return false;
        }
        if (!HardwareLayouts[stage])
        {
            VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&HardwareSets[stage];
            if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&HardwareLayouts[stage]),"Create hardware schedule layout")) return false;
        }
    }
    for (unsigned pass=0;pass<HardwareKernels.size();++pass) if (!HardwareKernels[pass])
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteHardwareCompute[pass],module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=HardwareLayouts[0]; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&desc,nullptr,&HardwareKernels[pass]); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create hardware schedule kernel")) return false;
    }
    for (unsigned i=0;i<HardwareModules.size();++i) if (!HardwareModules[i])
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteHardwareGraphics[i],HardwareModules[i],LastError)) return false;
    if (!HardwareLoadPass)
    {
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
        if (!Check(CreateRenderPassFunction(Device,&desc,nullptr,&HardwareLoadPass),"Create hardware ownership load pass")) return false;
    }
    const unsigned bins=unsigned(HardwarePlan.Bins.size()),setCount=bins+2;
    const auto alignment=std::max<VkDeviceSize>(1,DeviceLimits.minUniformBufferOffsetAlignment);
    const auto stride=(sizeof(NaniteHardwareDrawConstants)+alignment-1)/alignment*alignment;
    if (!CreateRawBuffer(stride*(bins+1),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Uniform)) return false;
    std::memcpy(work->Uniform.Mapped,&c,sizeof(c));
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,setCount},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,setCount*(MeshStagesEnabled?5:4)},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,bins*2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=setCount; pool.poolSizeCount=3; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create hardware schedule pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(setCount,HardwareSets[1]); layouts[0]=layouts[1]=HardwareSets[0]; std::vector<VkDescriptorSet> sets(setCount);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocate.descriptorPool=work->Pool; allocate.descriptorSetCount=setCount; allocate.pSetLayouts=layouts.data();
    if (!Check(AllocateDescriptorSets(Device,&allocate,sets.data()),"Allocate hardware schedule sets")) return false;
    const auto describe=[](const RawBuffer& b) { return VkDescriptorBufferInfo{b.Buffer,0,b.Size}; };
    for (unsigned i=0;i<setCount;++i)
    {
        const bool graphics=i>=2; VkDescriptorBufferInfo buffers[6]{}; VkDescriptorImageInfo images[2]{}; VkWriteDescriptorSet writes[8]{};
        buffers[0]={work->Uniform.Buffer,graphics?(i-1)*stride:0,graphics?sizeof(NaniteHardwareDrawConstants):sizeof(NaniteHardwareConstants)};
        if (graphics)
        {
            const auto& bin=HardwarePlan.Bins[i-2]; std::memcpy(static_cast<std::uint8_t*>(work->Uniform.Mapped)+(i-1)*stride,&bin.Constants,sizeof(bin.Constants));
            buffers[1]=describe(VisibilityGeometry); buffers[2]=describe(NaniteRasterFrameWork->Input->Allocation); buffers[3]=describe(work->Queue->Allocation); buffers[4]=describe(work->Owners->Allocation); buffers[5]=describe(work->Arguments->Allocation);
            const TextureHandle handles[]{bin.Diffuse,bin.Opacity};
            for (unsigned j=0;j<2;++j)
            {
                VkImageView view=DummyColor.View;
                if (handles[j])
                { const auto found=Textures.find(handles[j].Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8 || found->second.Image.Layout!=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return Fail("Invalid Vulkan hardware texture"); view=found->second.Image.View; }
                images[j]={ShadowSampler,view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }
        else
        {
            buffers[1]=describe(i?work->Ranges->Allocation:work->Owners->Allocation); buffers[2]=describe(NaniteRasterFrameWork->Output->Allocation);
            buffers[3]=describe(work->Arguments->Allocation); buffers[4]=describe(work->Queue->Allocation);
        }
        const unsigned count=graphics?(MeshStagesEnabled?8:7):5;
        for (unsigned binding=0;binding<count;++binding)
        {
            auto& write=writes[binding]; write={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=sets[i]; write.dstBinding=binding; write.descriptorCount=1;
            write.descriptorType=binding==0?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:(binding<5 || binding==7)?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (binding<5 || binding==7) write.pBufferInfo=&buffers[binding==7?5:binding]; else write.pImageInfo=&images[binding-5];
        }
        UpdateDescriptorSets(Device,count,writes,0,nullptr);
    }
    work->ComputeSets={sets[0],sets[1]}; work->GraphicsSets.assign(sets.begin()+2,sets.end()); HardwareWork=std::move(work); return true;
}
bool DrawNaniteHardware(const VisibilityBufferDesc& frame)
{
    if (!HardwarePlan.Active()) return true;
    auto& work=*HardwareWork; const auto& c=HardwarePlan.Constants;
    const RawBuffer* buffers[]{&work.Arguments->Allocation,&work.Queue->Allocation,&work.Owners->Allocation};
    const VkFlags shaderStages=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|(MeshStagesEnabled?VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT:0))|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkBufferMemoryBarrier barriers[3]{};
    for (unsigned i=0;i<3;++i)
    {
        auto& barrier=barriers[i]; barrier={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=buffers[i]->Buffer; barrier.size=buffers[i]->Size; barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    }
    CmdPipelineBarrier(CommandBuffer,shaderStages|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,3,barriers,0,nullptr);
    for (unsigned pass=0;pass<HardwareKernels.size();++pass)
    {
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,HardwareLayouts[0],0,1,&work.ComputeSets[pass?1:0],0,nullptr);
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,HardwareKernels[pass]);
        const unsigned items=pass==0?std::max(c.Counts.W,HardwarePlan.ArgumentWords):c.Counts.X;
        const unsigned rows=(items+NaniteHardwareThreads*NaniteHardwareColumns-1)/(NaniteHardwareThreads*NaniteHardwareColumns);
        NaniteGeometryDispatch(CommandBuffer,pass==2?1:NaniteHardwareColumns,pass==2?1:std::max(1u,rows),1);
        for (auto& barrier:barriers) barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,3,barriers,0,nullptr);
    }
    barriers[0].dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT|(frame.MeshShaders?VK_ACCESS_SHADER_READ_BIT:0); barriers[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,shaderStages|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,0,nullptr,3,barriers,0,nullptr);
    if (frame.MeshShaders) BuildMeshDispatchArguments();
    for (unsigned phase=0;phase<3;++phase)
    {
        if (phase==2)
        {
            barriers[2].srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barriers[2].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,1,&barriers[2],0,nullptr);
        }
        for (auto& image:VisibilityImages) TransitionImage(CommandBuffer,image,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        TransitionImage(CommandBuffer,MainDepth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        VkClearValue clear[3]{}; VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; pass.renderPass=phase || NaniteRasterPost?HardwareLoadPass:VisibilityRenderPass; pass.framebuffer=VisibilityFramebuffer;
        pass.renderArea={{0,0},{MainColor.Width,MainColor.Height}}; pass.clearValueCount=3; pass.pClearValues=clear; CmdBeginRenderPass(CommandBuffer,&pass,VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0,0,float(MainColor.Width),float(MainColor.Height),0,1}; const VkRect2D scissor{{0,0},{MainColor.Width,MainColor.Height}};
        CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&scissor);
        for (unsigned index=0;index<(phase==2?1u:unsigned(HardwarePlan.Bins.size()));++index)
        {
            const auto& bin=HardwarePlan.Bins[index]; const auto pipeline=NaniteHardwarePipeline(phase,bin.Cull,bin.Mirrored,frame.MeshShaders && phase!=2);
            if (!pipeline) { CmdEndRenderPass(CommandBuffer); return false; }
            CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
            CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,HardwareLayouts[1],0,1,&work.GraphicsSets[index],0,nullptr);
            if (phase==2) CmdDraw(CommandBuffer,3,1,0,0);
            else if (frame.MeshShaders) DrawMeshTasksIndirect(CommandBuffer,MeshWork->Arguments->Allocation.Buffer,index*sizeof(MeshShaderGrid),1,sizeof(MeshShaderGrid));
            else HardwareDrawIndirect(CommandBuffer,work.Arguments->Allocation.Buffer,index*NaniteHardwareBinWords*sizeof(unsigned),1,NaniteHardwareBinWords*sizeof(unsigned));
        }
        CmdEndRenderPass(CommandBuffer);
    }
    auto& stats=PendingFrame.Statistics; stats.NaniteHardwareBinCount+=unsigned(HardwarePlan.Bins.size()); stats.NaniteHardwareDispatchCount+=unsigned(HardwareKernels.size());
    const auto dispatches=unsigned(HardwarePlan.Bins.size()*2);
    if (frame.MeshShaders)
    {
        stats.MeshShaderDispatchCount+=dispatches;
        stats.MeshShaderIndirectCount+=dispatches; ++stats.MeshShaderArgumentDispatchCount;
    }
    { stats.NaniteHardwareIndirectCount+=dispatches; stats.IndirectDrawCallCount+=dispatches; }
    return true;
}
