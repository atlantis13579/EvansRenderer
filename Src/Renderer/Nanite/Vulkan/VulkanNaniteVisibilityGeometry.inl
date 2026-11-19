// Each source has immutable descriptors and aligned transform constants. Decoded
// storage is read on the graphics/compute queue; the assembled table is device local.
struct NaniteAssemblySource
{
    NaniteVisibilityGeometryConstants Constants; unsigned GroupsX=0,GroupsY=0;
    BufferHandle Vertex,Index; VkDescriptorSet Set=0;
};
struct NaniteAssemblyResources
{
    Impl& Owner; RawBuffer Uniform,Table; VkDescriptorPool Pool=0;
    std::vector<NaniteAssemblySource> Sources;
    explicit NaniteAssemblyResources(Impl& owner):Owner(owner) {}
    ~NaniteAssemblyResources()
    { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); Owner.DestroyRawBuffer(Uniform); Owner.DestroyRawBuffer(Table); }
};
std::shared_ptr<NaniteAssemblyResources> NaniteAssemblyWork;
VkDescriptorSetLayout NaniteAssemblySetLayout=0;
VkPipelineLayout NaniteAssemblyLayout=0;
VkPipeline NaniteAssemblyPipeline=0;
void DestroyNaniteVisibilityGeometry()
{ DestroyNaniteInstances();
    NaniteAssemblyWork.reset();
    if (NaniteAssemblyPipeline) DestroyPipelineFunction(Device,NaniteAssemblyPipeline,nullptr);
    if (NaniteAssemblyLayout) DestroyPipelineLayoutFunction(Device,NaniteAssemblyLayout,nullptr);
    if (NaniteAssemblySetLayout) DestroyDescriptorSetLayoutFunction(Device,NaniteAssemblySetLayout,nullptr);
    NaniteAssemblyPipeline=NaniteAssemblyLayout=NaniteAssemblySetLayout=0;
}
bool PrepareNaniteVisibilityGeometry(const VisibilityScene& scene)
{
    if (!GraphicsQueueHasCompute || DeviceLimits.maxPerStageDescriptorStorageBuffers<3 || DeviceLimits.maxDescriptorSetStorageBuffers<3 ||
        DeviceLimits.maxComputeWorkGroupInvocations<NaniteRasterThreads || DeviceLimits.maxComputeWorkGroupSize[0]<NaniteRasterThreads ||
        DeviceLimits.maxUniformBufferRange<sizeof(NaniteVisibilityGeometryConstants)) return Fail("Vulkan native visibility assembly exceeds compute limits");
    if (!ValidateVisibilityScene(scene,LastError)) return false;
    const auto bytes=scene.TriangleCount()*sizeof(VisibilityTriangle);
    if (bytes>DeviceLimits.maxStorageBufferRange) return Fail("Vulkan native visibility table exceeds storage limits");
    auto work=std::make_shared<NaniteAssemblyResources>(*this); work->Sources.reserve(scene.NativeMeshes.size());
    for (unsigned i=0;i<scene.NativeMeshes.size();++i)
    {
        const auto& mesh=scene.NativeMeshes[i]; const auto vertex=Buffers.find(mesh.VertexBuffer.Value),index=Buffers.find(mesh.IndexBuffer.Value);
        if (vertex==Buffers.end() || index==Buffers.end() || !vertex->second.NaniteGeometry || !index->second.NaniteGeometry)
            return Fail("Vulkan visibility assembly requires native decoded buffers");
        NaniteAssemblySource source; source.Vertex=mesh.VertexBuffer; source.Index=mesh.IndexBuffer;
        if (!BuildNaniteVisibilityGeometryConstants(scene,i,vertex->second.Description,index->second.Description,source.Constants,source.GroupsX,source.GroupsY,LastError)) return false;
        if (source.GroupsX>DeviceLimits.maxComputeWorkGroupCount[0] || source.GroupsY>DeviceLimits.maxComputeWorkGroupCount[1] ||
            vertex->second.Description.SizeInBytes>DeviceLimits.maxStorageBufferRange || index->second.Description.SizeInBytes>DeviceLimits.maxStorageBufferRange)
            return Fail("Vulkan visibility assembly exceeds storage/dispatch limits");
        work->Sources.push_back(source);
    }
    if (!LoadDevice(NaniteGeometryCreatePipelines,"vkCreateComputePipelines") || !LoadDevice(NaniteGeometryDispatch,"vkCmdDispatch")) return false;
    if (!NaniteAssemblySetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (unsigned i=0;i<4;++i) bindings[i]={i,i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=4; desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&NaniteAssemblySetLayout),"Create Nanite assembly descriptors")) return false;
    }
    if (!NaniteAssemblyLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&NaniteAssemblySetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&NaniteAssemblyLayout),"Create Nanite assembly layout")) return false;
    }
    if (!NaniteAssemblyPipeline)
    {
        VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanNaniteVisibilityGeometry,module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=NaniteAssemblyLayout; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&desc,nullptr,&NaniteAssemblyPipeline); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create Nanite assembly pipeline")) return false;
    }
    const unsigned count=unsigned(work->Sources.size()); const auto alignment=std::max<VkDeviceSize>(1,DeviceLimits.minUniformBufferOffsetAlignment);
    const auto stride=(sizeof(NaniteVisibilityGeometryConstants)+alignment-1)/alignment*alignment;
    if (!CreateRawBuffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,work->Table) ||
        !CreateRawBuffer(stride*count,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,work->Uniform)) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,count},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,count*3}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=count; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create Nanite assembly pool")) return false;
    std::vector<VkDescriptorSetLayout> layouts(count,NaniteAssemblySetLayout); std::vector<VkDescriptorSet> sets(count);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Pool; allocation.descriptorSetCount=count; allocation.pSetLayouts=layouts.data();
    if (!Check(AllocateDescriptorSets(Device,&allocation,sets.data()),"Allocate Nanite assembly sets")) return false;
    for (unsigned i=0;i<count;++i)
    {
        auto& source=work->Sources[i]; source.Set=sets[i]; std::memcpy(static_cast<std::uint8_t*>(work->Uniform.Mapped)+i*stride,&source.Constants,sizeof(source.Constants));
        const auto& vertex=Buffers.at(source.Vertex.Value).Allocation; const auto& index=Buffers.at(source.Index.Value).Allocation;
        VkDescriptorBufferInfo buffers[]{{work->Uniform.Buffer,i*stride,sizeof(source.Constants)},{vertex.Buffer,0,vertex.Size},{index.Buffer,0,index.Size},{work->Table.Buffer,0,work->Table.Size}};
        VkWriteDescriptorSet writes[4]{};
        for (unsigned binding=0;binding<4;++binding)
        {
            auto& write=writes[binding]; write={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=source.Set; write.dstBinding=binding; write.descriptorCount=1;
            write.descriptorType=binding?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffers[binding];
        }
        UpdateDescriptorSets(Device,4,writes,0,nullptr);
    }
    DestroyRawBuffer(VisibilityGeometry); VisibilityGeometry=work->Table; work->Table={};
    NaniteAssemblyWork=std::move(work); UploadedVisibility.reset(); return true;
}
bool DrawNaniteVisibilityGeometry()
{
    auto& work=*NaniteAssemblyWork; NaniteCopyLocks.reserve(NaniteCopyLocks.size()+work.Sources.size()*2);
    for (const auto& source:work.Sources) { NaniteCopyLocks.push_back(source.Vertex); NaniteCopyLocks.push_back(source.Index); }
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteAssemblyPipeline);
    for (const auto& source:work.Sources)
    {
        VkBufferMemoryBarrier inputs[2]{}; const BufferHandle handles[]{source.Vertex,source.Index};
        for (unsigned i=0;i<2;++i)
        {
            const auto& buffer=Buffers.at(handles[i].Value).Allocation; auto& barrier=inputs[i]; barrier={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=buffer.Buffer; barrier.size=buffer.Size;
        }
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,2,inputs,0,nullptr);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,NaniteAssemblyLayout,0,1,&source.Set,0,nullptr);
        NaniteGeometryDispatch(CommandBuffer,source.GroupsX,source.GroupsY,1); ++PendingFrame.Statistics.NaniteGeometryAssemblyCount;
    }
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=VisibilityGeometry.Buffer; barrier.size=VisibilityGeometry.Size;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|(MeshStagesEnabled?VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT:0))|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr); return true;
}
