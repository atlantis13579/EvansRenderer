bool LumenHardwareEnabled=false;
VkPhysicalDeviceAccelerationStructurePropertiesKHR LumenHardwareLimits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
PFN_vkGetBufferDeviceAddress LumenBufferAddress=nullptr;
PFN_vkCreateAccelerationStructureKHR LumenCreateAcceleration=nullptr;
PFN_vkDestroyAccelerationStructureKHR LumenDestroyAcceleration=nullptr;
PFN_vkGetAccelerationStructureDeviceAddressKHR LumenAccelerationAddress=nullptr;
PFN_vkGetAccelerationStructureBuildSizesKHR LumenAccelerationSizes=nullptr;
PFN_vkCmdBuildAccelerationStructuresKHR LumenBuildAcceleration=nullptr;
VkDescriptorSetLayout LumenHardwareSetLayout=0;
void QueryLumenHardwareSupport()
{
    LumenHardwareEnabled=false;
    if (InstanceApiVersion<MakeVersion(1,2,0) || DeviceApiVersion<MakeVersion(1,2,0) || !GraphicsQueueHasCompute ||
        DeviceLimits.maxBoundDescriptorSets<2 || DeviceLimits.maxPerStageDescriptorStorageBuffers<8 || DeviceLimits.maxDescriptorSetStorageBuffers<8) return;
    const auto enumerate=reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(GetInstanceProcAddress(Instance,"vkEnumerateDeviceExtensionProperties"));
    const auto features=reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(GetInstanceProcAddress(Instance,"vkGetPhysicalDeviceFeatures2"));
    const auto properties=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(GetInstanceProcAddress(Instance,"vkGetPhysicalDeviceProperties2"));
    if (!enumerate || !features || !properties) return;
    std::uint32_t count=0; if (enumerate(PhysicalDevice,nullptr,&count,nullptr)!=VK_SUCCESS) return;
    std::vector<VkExtensionProperties> extensions(count); if (count && enumerate(PhysicalDevice,nullptr,&count,extensions.data())!=VK_SUCCESS) return;
    for (auto required:{"VK_KHR_acceleration_structure","VK_KHR_deferred_host_operations","VK_KHR_ray_query"})
        if (std::none_of(extensions.begin(),extensions.end(),[&](const auto& e) { return std::strcmp(e.extensionName,required)==0; })) return;
    VkPhysicalDeviceRayQueryFeaturesKHR query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,&query};
    VkPhysicalDeviceBufferDeviceAddressFeatures address{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,&acceleration};
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&address}; features(PhysicalDevice,&supported);
    if (!address.bufferDeviceAddress || !acceleration.accelerationStructure || !query.rayQuery) return;
    struct PropertiesStorage { VkStructureType Type; void* Next; alignas(8) std::array<std::byte,4096> Base; };
    PropertiesStorage limits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,&LumenHardwareLimits,{}}; properties(PhysicalDevice,&limits);
    const auto& p=LumenHardwareLimits; VkFormatProperties format{}; GetPhysicalDeviceFormatProperties(PhysicalDevice,VK_FORMAT_R32G32B32_SFLOAT,&format);
    constexpr VkFlags AccelerationVertexFormat=0x20000000;
    LumenHardwareEnabled=p.maxGeometryCount && p.maxInstanceCount && p.maxPrimitiveCount && p.maxPerStageDescriptorAccelerationStructures &&
        p.maxDescriptorSetAccelerationStructures && p.minAccelerationStructureScratchOffsetAlignment &&
        !(p.minAccelerationStructureScratchOffsetAlignment&(p.minAccelerationStructureScratchOffsetAlignment-1)) && (format.bufferFeatures&AccelerationVertexFormat)!=0;
}
struct LumenHardwareStructure { VkAccelerationStructureKHR Handle=0; RawBuffer Storage; };
struct LumenHardwareRecord
{
    Impl& Owner; LumenHardwareStructure Top; std::vector<LumenHardwareStructure> Bottom; RawBuffer Words;
    VkDescriptorPool Pool=0; VkDescriptorSet Set=0;
    explicit LumenHardwareRecord(Impl& owner):Owner(owner) {}
    ~LumenHardwareRecord()
    {
        if (Pool) Owner.DestroyDescriptorPoolFunction(Owner.Device,Pool,nullptr);
        if (Top.Handle) Owner.LumenDestroyAcceleration(Owner.Device,Top.Handle,nullptr); Owner.DestroyRawBuffer(Top.Storage);
        for (auto& value:Bottom) { if (value.Handle) Owner.LumenDestroyAcceleration(Owner.Device,value.Handle,nullptr); Owner.DestroyRawBuffer(value.Storage); }
        Owner.DestroyRawBuffer(Words);
    }
};
enum class LumenHardwareProgram:unsigned { Distance,Radiance,Probes,Direct,Bounce,Visibility,Count };
std::array<VkPipeline,unsigned(LumenHardwareProgram::Count)> LumenHardwarePipelines{};
void DestroyLumenHardwarePrograms()
{
    for (auto pipeline:LumenHardwarePipelines) if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); LumenHardwarePipelines={};
}
bool CreateLumenQueryLayout(VkDescriptorSetLayout base,VkPipelineLayout& result)
{
    if (!EnsureLumenHardwareAPI()) return false;
    const VkDescriptorSetLayout sets[]{base,LumenHardwareSetLayout};
    VkPipelineLayoutCreateInfo create{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; create.setLayoutCount=LumenHardwareEnabled?2u:1u; create.pSetLayouts=sets;
    return Check(CreatePipelineLayoutFunction(Device,&create,nullptr,&result),"Create Lumen transport pipeline layout");
}
bool EnsureLumenHardwarePipeline(LumenHardwareProgram program,VkPipelineLayout layout)
{
    auto& pipeline=LumenHardwarePipelines[unsigned(program)]; if (pipeline) return true;
    const auto& shader=VulkanLumenHardwareShaders[unsigned(program)];
    VkShaderModule module=0; if (!CreateVulkanShaderModule(Device,CreateShaderModuleFunction,shader,module,LastError)) return false;
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; create.layout=layout; create.basePipelineIndex=-1;
    create.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",nullptr};
    const auto result=Evans::Renderer::CreateCachedVulkanComputePipelines(DistanceCreateCompute,Device,0,1,&create,nullptr,&pipeline); DestroyShaderModuleFunction(Device,module,nullptr);
    return Check(result,"Create Lumen inline-ray pipeline");
}
void BindLumenHardware(VkCommandBuffer commands,VkPipelineLayout layout,const LumenDistanceRecord& distance)
{
    if (distance.Hardware) CmdBindDescriptorSets(commands,VK_PIPELINE_BIND_POINT_COMPUTE,layout,1,1,&distance.Hardware->Set,0,nullptr);
}
bool EnsureLumenHardwareAPI()
{
    if (!LumenHardwareEnabled) return true;
    if (!LoadDevice(LumenBufferAddress,"vkGetBufferDeviceAddress") || !LoadDevice(LumenCreateAcceleration,"vkCreateAccelerationStructureKHR") ||
        !LoadDevice(LumenDestroyAcceleration,"vkDestroyAccelerationStructureKHR") || !LoadDevice(LumenAccelerationAddress,"vkGetAccelerationStructureDeviceAddressKHR") ||
        !LoadDevice(LumenAccelerationSizes,"vkGetAccelerationStructureBuildSizesKHR") || !LoadDevice(LumenBuildAcceleration,"vkCmdBuildAccelerationStructuresKHR")) return false;
    if (!LumenHardwareSetLayout)
    {
        const VkDescriptorSetLayoutBinding bindings[]{{0,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; create.bindingCount=2; create.pBindings=bindings;
        if (!Check(CreateDescriptorSetLayoutFunction(Device,&create,nullptr,&LumenHardwareSetLayout),"Create Lumen hardware query layout")) return false;
    }
    return true;
}
VkDeviceAddress LumenHardwareBufferAddress(const RawBuffer& buffer)
{ const VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,nullptr,buffer.Buffer}; return LumenBufferAddress(Device,&info); }
bool LumenHardwareStorage(VkDeviceSize bytes,std::int32_t type,LumenHardwareStructure& result)
{
    if (!bytes || !CreateRawBuffer(bytes,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,result.Storage)) return false;
    VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; create.buffer=result.Storage.Buffer; create.size=bytes; create.type=type;
    return Check(LumenCreateAcceleration(Device,&create,nullptr,&result.Handle),"Create Lumen hardware acceleration structure");
}
// Work owns all host input/scratch allocations until the same fence that owns
// the distance upload completes. AS records then retain only traversal storage.
bool BuildLumenHardwareAcceleration(LumenDistanceWork& work)
{
    const auto& snapshot=work.Record->Info.Hardware; if (!snapshot || snapshot->Instances.empty() || !LumenHardwareEnabled) return true;
    if (!EnsureLumenHardwareAPI()) return false;
    if (snapshot->Instances.size()>LumenHardwareLimits.maxInstanceCount ||
        std::max(std::size_t{1},snapshot->Words.size())*sizeof(unsigned)>DeviceLimits.maxStorageBufferRange) return Fail("Lumen hardware scene exceeds native instance/storage limits");
    const auto input=[&](const void* data,std::size_t bytes,VkFlags usage,RawBuffer& buffer) {
        if (!CreateRawBuffer(bytes,usage,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,buffer)) return false;
        std::memcpy(buffer.Mapped,data,bytes); work.HardwareInputs.push_back(buffer); return true;
    };
    auto record=std::make_shared<LumenHardwareRecord>(*this); work.Record->Hardware=record;
    const unsigned empty=0; RawBuffer wordInput; const auto wordBytes=std::max(std::size_t{1},snapshot->Words.size())*sizeof(unsigned);
    if (!input(snapshot->Words.empty()?&empty:snapshot->Words.data(),wordBytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,wordInput) ||
        !CreateRawBuffer(wordBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,record->Words)) return false;
    LumenDistanceBarrier(wordInput,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{0,0,wordBytes}; DistanceCopyBuffer(CommandBuffer,wordInput.Buffer,record->Words.Buffer,1,&copy);
    LumenDistanceBarrier(record->Words,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    std::vector<VkAccelerationStructureGeometryKHR> geometry(snapshot->Meshes.size());
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(snapshot->Meshes.size()); record->Bottom.resize(snapshot->Meshes.size());
    VkDeviceSize scratchSize=0; constexpr VkFlags vertexUsage=VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    for (unsigned mesh=0;mesh<snapshot->Meshes.size();++mesh)
    {
        const auto& triangles=snapshot->Meshes[mesh]->Triangles;
        if (triangles.size()>LumenHardwareLimits.maxPrimitiveCount) return Fail("Lumen hardware mesh exceeds the native primitive limit");
        std::vector<Float3> vertices; vertices.reserve(triangles.size()*3);
        for (const auto& triangle:triangles) for (auto p:{triangle.A,triangle.B,triangle.C}) vertices.push_back({p.X,p.Y,p.Z});
        RawBuffer buffer; if (!input(vertices.data(),vertices.size()*sizeof(Float3),vertexUsage,buffer)) return false;
        auto& g=geometry[mesh]; g.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR; g.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        auto& t=g.geometry.triangles; t.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR; t.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT;
        t.vertexData.deviceAddress=LumenHardwareBufferAddress(buffer); t.vertexStride=sizeof(Float3); t.maxVertex=unsigned(vertices.size()-1); t.indexType=VK_INDEX_TYPE_NONE_KHR;
        if (!t.vertexData.deviceAddress) return Fail("Lumen hardware vertex buffer has no device address");
        LumenDistanceBarrier(buffer,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
        VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR}; build.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; build.geometryCount=1; build.pGeometries=&g;
        sizes[mesh].sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR; const unsigned count=unsigned(triangles.size());
        LumenAccelerationSizes(Device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&build,&count,&sizes[mesh]); scratchSize=std::max(scratchSize,sizes[mesh].buildScratchSize);
        if (!LumenHardwareStorage(sizes[mesh].accelerationStructureSize,build.type,record->Bottom[mesh])) return false;
    }
    std::vector<VkAccelerationStructureInstanceKHR> instances(std::max(std::size_t{1},snapshot->Instances.size()));
    constexpr unsigned GeometryMask=1,ShadowMask=2,MaskShift=24;
    for (unsigned index=0;index<snapshot->Instances.size();++index)
    {
        const auto& source=snapshot->Instances[index]; auto& instance=instances[index]; std::memcpy(instance.transform,source.World.Values.data(),sizeof(instance.transform));
        instance.customIndexMask=index|((GeometryMask|(source.CastShadow?ShadowMask:0))<<MaskShift);
        instance.sbtOffsetFlags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR<<MaskShift;
        const VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,nullptr,record->Bottom[source.Mesh].Handle};
        instance.accelerationStructureReference=LumenAccelerationAddress(Device,&address);
        if (!instance.accelerationStructureReference) return Fail("Lumen BLAS has no device address");
    }
    RawBuffer instanceInput; if (!input(instances.data(),instances.size()*sizeof(instances[0]),vertexUsage,instanceInput)) return false;
    LumenDistanceBarrier(instanceInput,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    VkAccelerationStructureGeometryKHR topGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; topGeometry.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topGeometry.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR; topGeometry.geometry.instances.data.deviceAddress=LumenHardwareBufferAddress(instanceInput);
    if (!topGeometry.geometry.instances.data.deviceAddress) return Fail("Lumen instance buffer has no device address");
    VkAccelerationStructureBuildGeometryInfoKHR top{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR}; top.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    top.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; top.geometryCount=1; top.pGeometries=&topGeometry;
    VkAccelerationStructureBuildSizesInfoKHR topSize{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR}; const unsigned count=unsigned(snapshot->Instances.size());
    LumenAccelerationSizes(Device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&top,&count,&topSize); scratchSize=std::max(scratchSize,topSize.buildScratchSize);
    const VkDeviceSize alignment=LumenHardwareLimits.minAccelerationStructureScratchOffsetAlignment; RawBuffer scratch;
    if (!scratchSize || !CreateRawBuffer(scratchSize+alignment-1,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,scratch)) return false;
    work.HardwareInputs.push_back(scratch); const auto address=LumenHardwareBufferAddress(scratch);
    if (!address) return Fail("Lumen scratch buffer has no device address");
    const auto aligned=(address+alignment-1)/alignment*alignment;
    if (!LumenHardwareStorage(topSize.accelerationStructureSize,top.type,record->Top)) return false;
    for (unsigned mesh=0;mesh<geometry.size();++mesh)
    {
        VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR}; build.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build.flags=top.flags; build.geometryCount=1; build.pGeometries=&geometry[mesh]; build.dstAccelerationStructure=record->Bottom[mesh].Handle; build.scratchData.deviceAddress=aligned;
        const VkAccelerationStructureBuildRangeInfoKHR range{unsigned(snapshot->Meshes[mesh]->Triangles.size()),0,0,0}; const auto* ranges=&range;
        LumenBuildAcceleration(CommandBuffer,1,&build,&ranges);
        LumenDistanceBarrier(record->Bottom[mesh].Storage,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        LumenDistanceBarrier(scratch,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR|VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR|VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
    }
    top.dstAccelerationStructure=record->Top.Handle; top.scratchData.deviceAddress=aligned; const VkAccelerationStructureBuildRangeInfoKHR range{count,0,0,0}; const auto* ranges=&range;
    LumenBuildAcceleration(CommandBuffer,1,&top,&ranges);
    LumenDistanceBarrier(record->Top.Storage,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkDescriptorPoolSize poolSizes[]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
    if (!Check(CreateDescriptorPoolFunction(Device,&pool,nullptr,&record->Pool),"Create Lumen hardware query pool")) return false;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; allocation.descriptorPool=record->Pool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&LumenHardwareSetLayout;
    if (!Check(AllocateDescriptorSets(Device,&allocation,&record->Set),"Allocate Lumen hardware query set")) return false;
    const VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,nullptr,1,&record->Top.Handle};
    const VkDescriptorBufferInfo words{record->Words.Buffer,0,record->Words.Size}; VkWriteDescriptorSet writes[2]{};
    for (unsigned slot=0;slot<2;++slot) { auto& w=writes[slot]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=record->Set; w.dstBinding=slot; w.descriptorCount=1; w.descriptorType=poolSizes[slot].type; }
    writes[0].pNext=&as; writes[1].pBufferInfo=&words; UpdateDescriptorSets(Device,2,writes,0,nullptr); return true;
}
