// Native integer visibility attachments and immutable draw descriptor sets.
// Explicit color-write -> fragment-read barriers publish the payload. Existing
// image journaling rolls back speculative layouts when a frame is discarded.
std::array<RawImage, 2> VisibilityImages;
RawBuffer VisibilityGeometry;
std::shared_ptr<const VisibilityScene> UploadedVisibility;
VkImageView VisibilityDepthView = 0;
VkFramebuffer VisibilityFramebuffer = 0;
VkRenderPass VisibilityRenderPass = 0;
VkDescriptorSetLayout VisibilitySetLayout = 0;
VkPipelineLayout VisibilityLayout = 0;
std::array<VkShaderModule, 2> VisibilityRasterModules{}, VisibilityResolveModules{};
std::unordered_map<std::uint32_t, VkPipeline> VisibilityPipelines;
VkPipeline VisibilityResolvePipeline = 0;
struct VisibilitySlot { RawBuffer Uniform; VkDescriptorPool Pool = 0; VkDescriptorSet Set = 0; };
std::vector<VisibilitySlot> VisibilitySlots;

void DestroyVisibilityTargets()
{
    if (VisibilityFramebuffer && DestroyFramebufferFunction) DestroyFramebufferFunction(Device, VisibilityFramebuffer, nullptr);
    VisibilityFramebuffer = 0; VisibilityDepthView = 0;
    for (auto& image : VisibilityImages) DestroyRawImage(image);
}
void DestroyVisibilityBuffer()
{
    DestroyMeshShaders();
    DestroyNaniteMaterials();
    DestroyNaniteFrameRaster();
    DestroyNaniteVisibilityGeometry();
    DestroyVisibilityTargets(); DestroyRawBuffer(VisibilityGeometry); UploadedVisibility.reset();
    for (auto& slot : VisibilitySlots)
    { if (slot.Pool) DestroyDescriptorPoolFunction(Device, slot.Pool, nullptr); DestroyRawBuffer(slot.Uniform); }
    VisibilitySlots.clear();
    for (auto& entry : VisibilityPipelines) if (entry.second) DestroyPipelineFunction(Device, entry.second, nullptr);
    VisibilityPipelines.clear();
    if (VisibilityResolvePipeline) DestroyPipelineFunction(Device, VisibilityResolvePipeline, nullptr);
    for (auto* group : {&VisibilityRasterModules, &VisibilityResolveModules})
        for (auto& module : *group) { if (module) DestroyShaderModuleFunction(Device, module, nullptr); module = 0; }
    if (VisibilityRenderPass) DestroyRenderPassFunction(Device, VisibilityRenderPass, nullptr);
    if (VisibilityLayout) DestroyPipelineLayoutFunction(Device, VisibilityLayout, nullptr);
    if (VisibilitySetLayout) DestroyDescriptorSetLayoutFunction(Device, VisibilitySetLayout, nullptr);
    VisibilityResolvePipeline = VisibilityRenderPass = VisibilityLayout = VisibilitySetLayout = 0;
}
bool SupportsVisibilityBuffer() const
{
    if (!Device || !DeferredSupported || DeviceLimits.maxUniformBufferRange < sizeof(VisibilityConstants)) return false;
    for (auto format : {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_SFLOAT})
    {
        VkFormatProperties properties{}; GetPhysicalDeviceFormatProperties(PhysicalDevice, format, &properties);
        const auto required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & required) != required) return false;
    }
    return true;
}
bool PrepareVisibilityBuffer(const std::shared_ptr<const VisibilityScene>& scene, std::size_t drawCount)
{
    if (!scene || !SupportsVisibilityBuffer()) return Fail("Vulkan visibility requires sampled integer/float targets");
    const auto bytes = std::max(scene->TriangleCount(), std::size_t{1}) * sizeof(VisibilityTriangle);
    if (bytes > DeviceLimits.maxStorageBufferRange) return Fail("Vulkan visibility table exceeds maxStorageBufferRange");
    if (scene->HasNativeGeometry())
    {
        if (!PrepareNaniteVisibilityGeometry(*scene)) return false;
    }
    else if (UploadedVisibility != scene)
    {
        if (!ValidateVisibilityScene(*scene, LastError)) return false;
        std::vector<VisibilityTriangle> bounds;
        if (!BuildMeshShaderBounds(*scene,bounds,LastError)) return false;
        const auto geometryBytes=scene->Triangles.size()*sizeof(VisibilityTriangle);
        const auto tableBytes=std::max(scene->Triangles.size()+bounds.size(),std::size_t{1})*sizeof(VisibilityTriangle);
        if (tableBytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan mesh bounds exceed maxStorageBufferRange");
        RawBuffer candidate;
        if (!CreateRawBuffer(tableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, candidate))
        { DestroyRawBuffer(candidate); return false; }
        if (geometryBytes) std::memcpy(candidate.Mapped,scene->Triangles.data(),geometryBytes); else std::memset(candidate.Mapped,0,tableBytes);
        if (!bounds.empty()) std::memcpy(static_cast<std::byte*>(candidate.Mapped)+geometryBytes,bounds.data(),bounds.size()*sizeof(VisibilityTriangle));
        DestroyRawBuffer(VisibilityGeometry); VisibilityGeometry = candidate; UploadedVisibility = scene;
    }
    if (!VisibilitySetLayout)
    {
        std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
        for (std::uint32_t binding = 0; binding < bindings.size(); ++binding)
            bindings[binding] = {binding, binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
                (binding == 1 || binding >= 7) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                (binding < 2 || binding >= 7) ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | (MeshStagesEnabled?VK_SHADER_STAGE_MESH_BIT_EXT:0) |
                    (MeshTaskStagesEnabled?VK_SHADER_STAGE_TASK_BIT_EXT:0) : VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        desc.bindingCount = static_cast<std::uint32_t>(bindings.size()); desc.pBindings = bindings.data();
        if (!Check(CreateDescriptorSetLayoutFunction(Device, &desc, nullptr, &VisibilitySetLayout), "vkCreateDescriptorSetLayout(visibility)")) return false;
    }
    if (!VisibilityLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount = 1; desc.pSetLayouts = &VisibilitySetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device, &desc, nullptr, &VisibilityLayout), "vkCreatePipelineLayout(visibility)")) return false;
    }
    if (!VisibilityRenderPass)
    {
        std::array<VkAttachmentDescription, 3> attachments{};
        for (std::size_t index = 0; index < attachments.size(); ++index)
        {
            auto& attachment = attachments[index]; const bool depth = index == 2;
            attachment.format = depth ? VK_FORMAT_D32_SFLOAT : index == 0 ? VK_FORMAT_R32_UINT : VK_FORMAT_R32G32_SFLOAT;
            attachment.samples = 1; attachment.loadOp = depth ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = attachment.finalLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        const VkAttachmentReference colors[] {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
        const VkAttachmentReference depth{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 2; subpass.pColorAttachments = colors; subpass.pDepthStencilAttachment = &depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass = VK_SUBPASS_EXTERNAL; dependency.dstSubpass = 0;
        dependency.srcStageMask = dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask = dependency.srcAccessMask | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo desc{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; desc.attachmentCount = 3; desc.pAttachments = attachments.data();
        desc.subpassCount = 1; desc.pSubpasses = &subpass; desc.dependencyCount = 1; desc.pDependencies = &dependency;
        if (!Check(CreateRenderPassFunction(Device, &desc, nullptr, &VisibilityRenderPass), "vkCreateRenderPass(visibility)")) return false;
    }
    if (!VisibilityFramebuffer || VisibilityImages[0].Width != MainColor.Width || VisibilityImages[0].Height != MainColor.Height || VisibilityDepthView != MainDepth.View)
    {
        DestroyVisibilityTargets(); const VkFormat formats[] {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_SFLOAT};
        for (std::size_t index = 0; index < 2; ++index)
            if (!CreateRawImage(MainColor.Width, MainColor.Height, formats[index], VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VisibilityImages[index])) return false;
        const VkImageView views[] {VisibilityImages[0].View, VisibilityImages[1].View, MainDepth.View};
        if (!CreateFramebuffer(VisibilityRenderPass, views, 3, MainColor.Width, MainColor.Height, VisibilityFramebuffer)) return false;
        VisibilityDepthView = MainDepth.View;
    }
    for (std::size_t stage = 0; stage < 2; ++stage)
        if (!VisibilityRasterModules[stage] && !CreateVulkanShaderModule(Device, CreateShaderModuleFunction,
            VulkanVisibilityBufferShaders[stage], VisibilityRasterModules[stage], LastError)) return false;
    if (!CreateFullscreenPipeline(&VulkanVisibilityBufferShaders[2], VisibilityResolveModules, DeferredRenderPasses[0],
        VisibilityResolvePipeline, VisibilityLayout, FullscreenBlend::Replace, false, 1, false, GBufferAttachmentCount)) return false;
    while (VisibilitySlots.size() < drawCount)
    {
        VisibilitySlot slot;
        if (!CreateRawBuffer(sizeof(VisibilityConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, slot.Uniform))
        { DestroyRawBuffer(slot.Uniform); return false; }
        const VkDescriptorPoolSize sizes[] {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets = 1; pool.poolSizeCount = 3; pool.pPoolSizes = sizes;
        if (!Check(CreateDescriptorPoolFunction(Device, &pool, nullptr, &slot.Pool), "vkCreateDescriptorPool(visibility draw)"))
        { DestroyRawBuffer(slot.Uniform); return false; }
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = slot.Pool; allocation.descriptorSetCount = 1; allocation.pSetLayouts = &VisibilitySetLayout;
        if (!Check(AllocateDescriptorSets(Device, &allocation, &slot.Set), "vkAllocateDescriptorSets(visibility draw)"))
        { DestroyDescriptorPoolFunction(Device, slot.Pool, nullptr); DestroyRawBuffer(slot.Uniform); return false; }
        try { VisibilitySlots.push_back(slot); }
        catch (...) { DestroyDescriptorPoolFunction(Device, slot.Pool, nullptr); DestroyRawBuffer(slot.Uniform); throw; }
    }
    return true;
}
VkPipeline VisibilityPipeline(const PipelineState& state, bool mirrored, bool mesh=false,bool task=false)
{
    const std::uint32_t key = (state.Fill == FillMode::Wireframe ? 1u : 0u) | (state.BackFaceCulling ? 2u : 0u) |
        (mirrored ? 4u : 0u) | (state.DepthTest ? 8u : 0u) | (state.DepthWrite ? 16u : 0u) | (mesh?32u:0u)|(task?64u:0u);
    const auto found = VisibilityPipelines.find(key); if (found != VisibilityPipelines.end()) return found->second;
    VkPipelineShaderStageCreateInfo stages[3]{};
    for (std::size_t stage = 0; stage < 2; ++stage)
    { stages[stage].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[stage].stage = stage ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT; stages[stage].module = VisibilityRasterModules[stage]; stages[stage].pName = "main"; }
    if (mesh)
    {
        stages[0].stage=VK_SHADER_STAGE_MESH_BIT_EXT; stages[0].module=MeshModules[task?2:0]; stages[1].module=MeshFragmentModules[0];
        if (task) stages[2]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_TASK_BIT_EXT,MeshTaskModule,"main",nullptr};
    }
    VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = state.Fill == FillMode::Wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    raster.cullMode = state.BackFaceCulling ? CullBackFaces : VK_CULL_MODE_NONE;
    raster.frontFace = mirrored ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; samples.rasterizationSamples = 1;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = state.DepthTest; depth.depthWriteEnable = state.DepthWrite;
    depth.depthCompareOp = state.DepthWrite ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_LESS_OR_EQUAL; depth.maxDepthBounds = 1;
    VkPipelineColorBlendAttachmentState colors[2]{};
    for (auto& color : colors) color.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; blend.attachmentCount = 2; blend.pAttachments = colors;
    const VkDynamicState dynamicStates[] {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dynamic.dynamicStateCount = 2; dynamic.pDynamicStates = dynamicStates;
    VkGraphicsPipelineCreateInfo desc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    desc.stageCount = mesh && task?3:2; desc.pStages = stages; desc.pVertexInputState = mesh?nullptr:&vertices; desc.pInputAssemblyState = mesh?nullptr:&assembly;
    desc.pViewportState = &viewport; desc.pRasterizationState = &raster; desc.pMultisampleState = &samples;
    desc.pDepthStencilState = &depth; desc.pColorBlendState = &blend; desc.pDynamicState = &dynamic;
    desc.layout = VisibilityLayout; desc.renderPass = VisibilityRenderPass;
    VkPipeline pipeline = 0;
    if (!Check(Evans::Renderer::CreateCachedVulkanGraphicsPipelines(CreateGraphicsPipelines,Device,0, 1, &desc, nullptr, &pipeline), "vkCreateGraphicsPipelines(visibility raster)")) return 0;
    VisibilityPipelines.emplace(key, pipeline); return pipeline;
}
bool DrawVisibilityBuffer(const VisibilityBufferDesc& frame)
{
    for (std::size_t index = 0; index < frame.Draws.size(); ++index)
    {
        const auto& draw = frame.Draws[index]; auto& slot = VisibilitySlots[index];
        const TextureHandle handles[] {draw.Pipeline.DiffuseTexture, draw.Pipeline.NormalTexture, draw.Pipeline.OpacityTexture};
        VkDescriptorImageInfo images[5] {{DeferredSampler, VisibilityImages[0].View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {DeferredSampler, VisibilityImages[1].View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        for (std::size_t image = 0; image < 3; ++image)
        {
            VkImageView view = DummyColor.View;
            if (handles[image])
            {
                const auto found = Textures.find(handles[image].Value);
                if (found == Textures.end() || found->second.Description.Format != TextureFormat::RGBA8 ||
                    found->second.Image.Layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return Fail("Vulkan visibility material texture is invalid");
                view = found->second.Image.View;
            }
            images[image + 2] = {MaterialSampler(), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        VisibilityConstants constants;
        if (!BuildVisibilityConstants(frame.ViewProjection, draw.Pipeline, CreateInfo.Width, CreateInfo.Height, frame.Scene->TriangleCount(),
            frame.Scene->Draws[draw.Range], bool(handles[0]), bool(handles[1]), bool(handles[2]), constants)) return Fail("Invalid Vulkan visibility constants");
        constants.MaterialTiles=MaterialPlan.Resolve(unsigned(index));
        std::memcpy(slot.Uniform.Mapped, &constants, sizeof(constants));
        const auto& mapping=MaterialPlan.Active()?MaterialMap->Allocation:VisibilityGeometry;
        const auto& work=MaterialPlan.Active()?MaterialWork->Allocation:VisibilityGeometry;
        const VkDescriptorBufferInfo buffers[] {{slot.Uniform.Buffer, 0, sizeof(constants)}, {VisibilityGeometry.Buffer, 0, VisibilityGeometry.Size},
            {mapping.Buffer,0,mapping.Size},{work.Buffer,0,work.Size}};
        VkWriteDescriptorSet writes[9]{};
        for (std::uint32_t binding = 0; binding < 9; ++binding)
        {
            auto& write = writes[binding]; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet = slot.Set;
            write.dstBinding = binding; write.descriptorCount = 1;
            write.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : (binding == 1 || binding >= 7) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (binding < 2 || binding >= 7) write.pBufferInfo = &buffers[binding>=7?binding-5:binding]; else write.pImageInfo = &images[binding - 2];
        }
        UpdateDescriptorSets(Device, 9, writes, 0, nullptr);
    }
    // End the initial G-buffer clear after validating every draw. A failed
    // material still leaves a native pass open for the ordinary abort path.
    CmdEndRenderPass(CommandBuffer);
    VkClearValue clear[3]{}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    if (frame.NaniteRaster) { if (!DrawNaniteFrameRaster(frame)) return false; }
    else
    {
        for (auto& image : VisibilityImages) TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        begin.renderPass = VisibilityRenderPass; begin.framebuffer = VisibilityFramebuffer; begin.renderArea = {{0, 0}, {MainColor.Width, MainColor.Height}};
        begin.clearValueCount = 3; begin.pClearValues = clear; CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{0, 0, float(MainColor.Width), float(MainColor.Height), 0, 1};
        const VkRect2D scissor{{0, 0}, {MainColor.Width, MainColor.Height}};
        CmdSetViewport(CommandBuffer, 0, 1, &viewport); CmdSetScissor(CommandBuffer, 0, 1, &scissor);
        for (std::size_t index = 0; index < frame.Draws.size(); ++index)
        {
            const auto& draw = frame.Draws[index]; const auto& range = frame.Scene->Draws[draw.Range];
            const bool task=frame.MeshShaders && UseMeshTaskStage(frame.MeshDispatch,range.TriangleCount,MeshTaskStagesEnabled);
            auto pipeline = VisibilityPipeline(draw.Pipeline, range.Mirrored,frame.MeshShaders,task); if (!pipeline) return false;
            CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            CmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VisibilityLayout, 0, 1, &VisibilitySlots[index].Set, 0, nullptr);
            if (frame.MeshShaders)
            {
                const auto grid=task?BuildMeshTaskGrid(range.TriangleCount):BuildMeshShaderGrid(range.TriangleCount);
                DrawMeshTasks(CommandBuffer,grid.X,grid.Y,grid.Z);
                if (task) ++PendingFrame.Statistics.MeshShaderObjectDispatchCount;
            }
            else CmdDraw(CommandBuffer,range.TriangleCount*VisibilityTriangleCorners,1,0,0);
        }
        CmdEndRenderPass(CommandBuffer);
    }
    if (!ClassifyNaniteMaterials()) return false;
    begin.renderArea = {{0, 0}, {MainColor.Width, MainColor.Height}};
    for (auto& image : VisibilityImages) TransitionImage(CommandBuffer, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    begin.renderPass = DeferredRenderPasses[0]; begin.framebuffer = GBufferFramebuffer; begin.clearValueCount = 0; begin.pClearValues = nullptr;
    CmdBeginRenderPass(CommandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    CmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VisibilityResolvePipeline);
    const auto resolveCount=MaterialPlan.Active()?MaterialPlan.Representatives.size():frame.Draws.size();
    for (unsigned bin=0;bin<resolveCount;++bin)
    {
        const auto index=MaterialPlan.Active()?MaterialPlan.Representatives[bin]:bin;
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VisibilityLayout,0,1,&VisibilitySlots[index].Set,0,nullptr);
        if (MaterialPlan.Active()) MaterialDrawIndirect(CommandBuffer,MaterialWork->Allocation.Buffer,bin*NaniteMaterialBinWords*sizeof(unsigned),1,4*sizeof(unsigned));
        else CmdDraw(CommandBuffer,3,1,0,0);
    }
    // Leave the final material render pass open for the ordinary EndPass path.
    return true;
}
