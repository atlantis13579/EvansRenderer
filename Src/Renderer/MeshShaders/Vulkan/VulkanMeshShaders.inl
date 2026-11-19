// EXT mesh stages are negotiated separately from the Vulkan 1.0 raster path.
// Mesh stages and primitive-input fragments target Vulkan 1.2/SPIR-V 1.5.
std::uint32_t InstanceApiVersion=MakeVersion(1,0,0),DeviceApiVersion=MakeVersion(1,0,0);
bool MeshStagesEnabled=false,MeshTaskStagesEnabled=false,MeshNativeBarycentrics=false;
PFN_vkCmdDrawMeshTasksEXT DrawMeshTasks=nullptr;
PFN_vkCmdDrawMeshTasksIndirectEXT DrawMeshTasksIndirect=nullptr;
std::array<VkShaderModule,3> MeshModules{}; // Direct, Nanite, task-fed ordinary mesh.
VkShaderModule MeshTaskModule=0;
std::array<VkShaderModule,3> MeshFragmentModules{};
VkDescriptorSetLayout MeshArgumentSetLayout=0;
VkPipelineLayout MeshArgumentLayout=0;
VkPipeline MeshArgumentPipeline=0;
struct MeshResources
{
    Impl& Owner; VkDescriptorPool Pool=0; VkDescriptorSet Set=0;
    std::shared_ptr<NaniteFrameBuffer> Arguments;
    explicit MeshResources(Impl& owner):Owner(owner) {}
    ~MeshResources() { if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr); }
};
std::shared_ptr<MeshResources> MeshWork;

void QueryMeshShaderSupport()
{
    MeshStagesEnabled=MeshTaskStagesEnabled=MeshNativeBarycentrics=false;
    if (InstanceApiVersion<MakeVersion(1,2,0) || DeviceApiVersion<MakeVersion(1,2,0) || !GraphicsQueueHasCompute ||
        DeviceLimits.maxPerStageDescriptorStorageBuffers<5 || DeviceLimits.maxDescriptorSetStorageBuffers<5) return;
    const auto enumerate=reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(GetInstanceProcAddress(Instance,"vkEnumerateDeviceExtensionProperties"));
    const auto getFeatures=reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(GetInstanceProcAddress(Instance,"vkGetPhysicalDeviceFeatures2"));
    const auto getProperties=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(GetInstanceProcAddress(Instance,"vkGetPhysicalDeviceProperties2"));
    if (!enumerate || !getFeatures || !getProperties) return;
    std::uint32_t count=0;
    if (enumerate(PhysicalDevice,nullptr,&count,nullptr)!=VK_SUCCESS) return;
    std::vector<VkExtensionProperties> extensions(count);
    if (enumerate(PhysicalDevice,nullptr,&count,extensions.data())!=VK_SUCCESS) return;
    bool found=false,barycentric=false;
    for (const auto& extension:extensions)
    {
        if (std::strcmp(extension.extensionName,"VK_EXT_mesh_shader")==0) found=true;
        if (std::strcmp(extension.extensionName,"VK_KHR_fragment_shader_barycentric")==0) barycentric=true;
    }
    if (!found) return;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&mesh}; getFeatures(PhysicalDevice,&features);
    if (!mesh.meshShader) return;
    VkPhysicalDeviceMeshShaderPropertiesEXT limits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    // The base device-properties payload uses the same oversized, aligned
    // storage as ordinary device selection; only the typed pNext is consumed.
    struct PropertiesStorage { VkStructureType Type; void* Next; alignas(8) std::array<std::byte,4096> Base; };
    PropertiesStorage properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,&limits,{}}; getProperties(PhysicalDevice,&properties);
    const auto maximum=BuildMeshShaderGrid(unsigned(VisibilityStorageCapacity));
    // Drivers round vertex and primitive storage independently. Include the
    // portable path's position/UV/alpha, planes/ID and triangle-index storage;
    // retain the previous padding floor rather than underbudget a new device.
    constexpr unsigned OutputPaddingBytes=8192,VertexWords=7,PrimitiveWords=13,OutputComponents=17,SharedBytes=MeshShaderSharedBytes;
    if (!limits.meshOutputPerVertexGranularity || !limits.meshOutputPerPrimitiveGranularity) return;
    const auto rounded=[](unsigned count,unsigned granularity)->std::uint64_t
    { return (std::uint64_t(count)+granularity-1)/granularity*granularity; };
    const auto outputBytes=(rounded(MeshShaderVertices,limits.meshOutputPerVertexGranularity)*VertexWords+
        rounded(MeshShaderTriangles,limits.meshOutputPerPrimitiveGranularity)*PrimitiveWords)*sizeof(unsigned);
    const auto outputByteBudget=std::max(std::uint64_t(OutputPaddingBytes),outputBytes);
    MeshStagesEnabled=limits.maxMeshWorkGroupInvocations>=MeshShaderThreads && limits.maxMeshWorkGroupSize[0]>=MeshShaderThreads &&
        limits.maxMeshWorkGroupCount[0]>=maximum.X && limits.maxMeshWorkGroupCount[1]>=maximum.Y &&
        limits.maxMeshWorkGroupTotalCount>=std::uint64_t(maximum.X)*maximum.Y && limits.maxMeshSharedMemorySize>=SharedBytes &&
        limits.maxMeshOutputVertices>=MeshShaderVertices && limits.maxMeshOutputPrimitives>=MeshShaderTriangles &&
        limits.maxMeshOutputMemorySize>=outputByteBudget && limits.maxMeshOutputComponents>=OutputComponents &&
        DeviceLimits.maxComputeWorkGroupInvocations>=MeshShaderThreads && DeviceLimits.maxComputeWorkGroupSize[0]>=MeshShaderThreads &&
        DeviceLimits.maxComputeWorkGroupCount[0]>=MeshShaderColumns;
    const auto taskMaximum=BuildMeshTaskGrid(unsigned(VisibilityStorageCapacity));
    constexpr unsigned TaskSharedBytes=sizeof(unsigned),PayloadBytes=MeshShaderPayloadBytes;
    MeshTaskStagesEnabled=MeshStagesEnabled && mesh.taskShader &&
        limits.maxTaskWorkGroupInvocations>=MeshShaderThreads && limits.maxTaskWorkGroupSize[0]>=MeshShaderThreads &&
        limits.maxTaskWorkGroupCount[0]>=taskMaximum.X && limits.maxTaskWorkGroupCount[1]>=taskMaximum.Y &&
        limits.maxTaskWorkGroupTotalCount>=std::uint64_t(taskMaximum.X)*taskMaximum.Y &&
        limits.maxTaskSharedMemorySize>=TaskSharedBytes && limits.maxTaskPayloadSize>=PayloadBytes &&
        limits.maxTaskPayloadAndSharedMemorySize>=PayloadBytes+TaskSharedBytes &&
        limits.maxMeshPayloadAndSharedMemorySize>=PayloadBytes+SharedBytes &&
        limits.maxMeshPayloadAndOutputMemorySize>=PayloadBytes+outputByteBudget;
    if (MeshStagesEnabled && barycentric)
    {
        VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&bary}; getFeatures(PhysicalDevice,&available);
        MeshNativeBarycentrics=bary.fragmentShaderBarycentric!=0;
    }
}
void DestroyMeshShaders()
{
    MeshWork.reset();
    for (auto module:MeshModules) if (module) DestroyShaderModuleFunction(Device,module,nullptr); MeshModules={};
    if (MeshTaskModule) DestroyShaderModuleFunction(Device,MeshTaskModule,nullptr); MeshTaskModule=0;
    for (auto module:MeshFragmentModules) if (module) DestroyShaderModuleFunction(Device,module,nullptr); MeshFragmentModules={};
    if (MeshArgumentPipeline) DestroyPipelineFunction(Device,MeshArgumentPipeline,nullptr); MeshArgumentPipeline=0;
    if (MeshArgumentLayout) DestroyPipelineLayoutFunction(Device,MeshArgumentLayout,nullptr); MeshArgumentLayout=0;
    if (MeshArgumentSetLayout) DestroyDescriptorSetLayoutFunction(Device,MeshArgumentSetLayout,nullptr); MeshArgumentSetLayout=0;
}
bool PrepareMeshShaders(const VisibilityBufferDesc& frame)
{
    MeshWork.reset(); if (!frame.MeshShaders) return true;
    if (!MeshStagesEnabled || !DrawMeshTasks || !DrawMeshTasksIndirect) return Fail("Vulkan EXT mesh stages are unavailable on this device");
    bool needed[3]{false,frame.NaniteRaster,false};
    if (!frame.NaniteRaster) for (const auto& draw:frame.Draws)
        needed[UseMeshTaskStage(frame.MeshDispatch,frame.Scene->Draws[draw.Range].TriangleCount,MeshTaskStagesEnabled)?2:0]=true;
    for (unsigned variant=0;variant<3;++variant) if (needed[variant] && !MeshModules[variant])
    {
        const unsigned program=variant==1?1:variant==2?(MeshNativeBarycentrics?6:5):(MeshNativeBarycentrics?3:0);
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanMeshShaders[program],MeshModules[variant],LastError)) return false;
    }
    if (needed[2] && !MeshTaskModule &&
        !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanMeshShaders[4],MeshTaskModule,LastError)) return false;
    for (unsigned fragment=frame.NaniteRaster?1:0;fragment<(frame.NaniteRaster?3u:1u);++fragment)
        if (!MeshFragmentModules[fragment] && !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanMeshFragments[!fragment && MeshNativeBarycentrics?3:fragment],MeshFragmentModules[fragment],LastError)) return false;
    if (!frame.NaniteRaster || !HardwarePlan.Active()) return true;
    const auto grid=BuildMeshShaderGrid(unsigned(HardwarePlan.Bins.size()));
    if (grid.Y>DeviceLimits.maxComputeWorkGroupCount[1]) return Fail("Mesh argument preparation exceeds the device dispatch limit");
    if (!MeshArgumentSetLayout)
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (unsigned i=0;i<3;++i) bindings[i]={i,i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; desc.bindingCount=3; desc.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&desc,nullptr,&MeshArgumentSetLayout),"Create mesh argument descriptors")) return false;
    }
    if (!MeshArgumentLayout)
    {
        VkPipelineLayoutCreateInfo desc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; desc.setLayoutCount=1; desc.pSetLayouts=&MeshArgumentSetLayout;
        if (!Check(CreatePipelineLayoutFunction(Device,&desc,nullptr,&MeshArgumentLayout),"Create mesh argument layout")) return false;
    }
    if (!MeshArgumentPipeline)
    {
        VkShaderModule module=0;
        if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,VulkanMeshShaders[2],module,LastError)) return false;
        VkComputePipelineCreateInfo desc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; desc.layout=MeshArgumentLayout; desc.basePipelineIndex=-1;
        desc.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
        const auto status=Evans::Renderer::CreateCachedVulkanComputePipelines(NaniteGeometryCreatePipelines,Device,0,1,&desc,nullptr,&MeshArgumentPipeline); DestroyShaderModuleFunction(Device,module,nullptr);
        if (!Check(status,"Create mesh argument pipeline")) return false;
    }
    auto work=std::make_shared<MeshResources>(*this);
    if (!CreateNaniteFrameBuffer(unsigned(HardwarePlan.Bins.size())*MeshShaderArgumentWords,nullptr,true,work->Arguments,true)) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&work->Pool),"Create mesh argument pool")) return false;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=work->Pool;
    allocation.descriptorSetCount=1; allocation.pSetLayouts=&MeshArgumentSetLayout;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&work->Set),"Allocate mesh argument set")) return false;
    const auto& source=HardwareWork->Arguments->Allocation; const auto& target=work->Arguments->Allocation;
    const VkDescriptorBufferInfo buffers[]{{HardwareWork->Uniform.Buffer,0,sizeof(NaniteHardwareConstants)},
        {source.Buffer,0,source.Size},{target.Buffer,0,target.Size}};
    VkWriteDescriptorSet writes[3]{};
    for (unsigned i=0;i<3;++i)
    {
        auto& write=writes[i]; write={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; write.dstSet=work->Set; write.dstBinding=i;
        write.descriptorCount=1; write.descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; write.pBufferInfo=&buffers[i];
    }
    UpdateDescriptorSets(Device,3,writes,0,nullptr); MeshWork=std::move(work); return true;
}
void BuildMeshDispatchArguments()
{
    auto& buffer=MeshWork->Arguments->Allocation;
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer=buffer.Buffer; barrier.size=buffer.Size; barrier.srcAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,MeshArgumentPipeline);
    CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,MeshArgumentLayout,0,1,&MeshWork->Set,0,nullptr);
    const auto grid=BuildMeshShaderGrid(unsigned(HardwarePlan.Bins.size())); NaniteGeometryDispatch(CommandBuffer,grid.X,grid.Y,grid.Z);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,0,nullptr,1,&barrier,0,nullptr);
}
