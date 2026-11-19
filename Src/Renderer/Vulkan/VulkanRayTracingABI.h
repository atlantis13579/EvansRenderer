#pragma once
#include "Renderer/Vulkan/VulkanABI.h"

namespace Evans::Renderer::VulkanABI
{
// KHR ray-query/build ABI, declared separately from the portable raster ABI.
// https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h
using VkDeviceAddress=std::uint64_t;
using VkAccelerationStructureKHR=std::uint64_t;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO=1000060000;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES=1000257000;
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO=1000244001;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR=1000150000;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR=1000150002;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR=1000150003;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR=1000150004;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR=1000150005;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR=1000150006;
constexpr VkStructureType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR=1000150007;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR=1000150013;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR=1000150014;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR=1000150017;
constexpr VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR=1000150020;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR=1000348013;
constexpr VkFlags VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT=2;
constexpr VkFlags VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT=0x00020000;
constexpr VkFlags VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR=0x00080000;
constexpr VkFlags VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR=0x00100000;
constexpr VkFlags VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR=0x00200000,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR=0x00400000;
constexpr VkFlags VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR=0x02000000;
constexpr VkFlags VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR=4;
constexpr VkFlags VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR=1;
constexpr VkIndexType VK_INDEX_TYPE_NONE_KHR=1000165000;
constexpr VkDescriptorType VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR=1000150000;
constexpr std::int32_t VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR=0,VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR=1;
constexpr std::int32_t VK_GEOMETRY_TYPE_TRIANGLES_KHR=0,VK_GEOMETRY_TYPE_INSTANCES_KHR=2;
constexpr std::int32_t VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR=0,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR=1;
struct VkPhysicalDeviceBufferDeviceAddressFeatures
{ VkStructureType sType; void* pNext; VkBool32 bufferDeviceAddress,bufferDeviceAddressCaptureReplay,bufferDeviceAddressMultiDevice; };
struct VkPhysicalDeviceAccelerationStructureFeaturesKHR
{ VkStructureType sType; void* pNext; VkBool32 accelerationStructure,accelerationStructureCaptureReplay,accelerationStructureIndirectBuild,accelerationStructureHostCommands,descriptorBindingAccelerationStructureUpdateAfterBind; };
struct VkPhysicalDeviceRayQueryFeaturesKHR { VkStructureType sType; void* pNext; VkBool32 rayQuery; };
struct VkPhysicalDeviceAccelerationStructurePropertiesKHR
{
    VkStructureType sType; void* pNext;
    std::uint64_t maxGeometryCount,maxInstanceCount,maxPrimitiveCount;
    std::uint32_t maxPerStageDescriptorAccelerationStructures,maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
    std::uint32_t maxDescriptorSetAccelerationStructures,maxDescriptorSetUpdateAfterBindAccelerationStructures,minAccelerationStructureScratchOffsetAlignment;
};
struct VkMemoryAllocateFlagsInfo { VkStructureType sType; const void* pNext; VkFlags flags; std::uint32_t deviceMask; };
struct VkBufferDeviceAddressInfo { VkStructureType sType; const void* pNext; VkBuffer buffer; };
union VkDeviceOrHostAddressConstKHR { VkDeviceAddress deviceAddress; const void* hostAddress; };
union VkDeviceOrHostAddressKHR { VkDeviceAddress deviceAddress; void* hostAddress; };
struct VkAccelerationStructureGeometryTrianglesDataKHR
{
    VkStructureType sType; const void* pNext; VkFormat vertexFormat; VkDeviceOrHostAddressConstKHR vertexData;
    VkDeviceSize vertexStride; std::uint32_t maxVertex; VkIndexType indexType; VkDeviceOrHostAddressConstKHR indexData,transformData;
};
struct VkAccelerationStructureGeometryAabbsDataKHR { VkStructureType sType; const void* pNext; VkDeviceOrHostAddressConstKHR data; VkDeviceSize stride; };
struct VkAccelerationStructureGeometryInstancesDataKHR { VkStructureType sType; const void* pNext; VkBool32 arrayOfPointers; VkDeviceOrHostAddressConstKHR data; };
union VkAccelerationStructureGeometryDataKHR
{ VkAccelerationStructureGeometryTrianglesDataKHR triangles; VkAccelerationStructureGeometryAabbsDataKHR aabbs; VkAccelerationStructureGeometryInstancesDataKHR instances; };
struct VkAccelerationStructureGeometryKHR
{ VkStructureType sType; const void* pNext; std::int32_t geometryType; VkAccelerationStructureGeometryDataKHR geometry; VkFlags flags; };
struct VkAccelerationStructureBuildGeometryInfoKHR
{
    VkStructureType sType; const void* pNext; std::int32_t type; VkFlags flags; std::int32_t mode;
    VkAccelerationStructureKHR srcAccelerationStructure,dstAccelerationStructure; std::uint32_t geometryCount;
    const VkAccelerationStructureGeometryKHR* pGeometries; const VkAccelerationStructureGeometryKHR* const* ppGeometries;
    VkDeviceOrHostAddressKHR scratchData;
};
struct VkAccelerationStructureCreateInfoKHR
{ VkStructureType sType; const void* pNext; VkFlags createFlags; VkBuffer buffer; VkDeviceSize offset,size; std::int32_t type; VkDeviceAddress deviceAddress; };
struct VkAccelerationStructureBuildSizesInfoKHR
{ VkStructureType sType; void* pNext; VkDeviceSize accelerationStructureSize,updateScratchSize,buildScratchSize; };
struct VkAccelerationStructureBuildRangeInfoKHR { std::uint32_t primitiveCount,primitiveOffset,firstVertex,transformOffset; };
struct VkAccelerationStructureDeviceAddressInfoKHR { VkStructureType sType; const void* pNext; VkAccelerationStructureKHR accelerationStructure; };
struct VkWriteDescriptorSetAccelerationStructureKHR
{ VkStructureType sType; const void* pNext; std::uint32_t accelerationStructureCount; const VkAccelerationStructureKHR* pAccelerationStructures; };
struct VkAccelerationStructureInstanceKHR
{
    float transform[3][4]; std::uint32_t customIndexMask,sbtOffsetFlags; std::uint64_t accelerationStructureReference;
};
static_assert(sizeof(VkAccelerationStructureInstanceKHR)==64,"Vulkan instance descriptor ABI changed");
using PFN_vkGetBufferDeviceAddress=VkDeviceAddress(EVANS_VK_CALL*)(VkDevice,const VkBufferDeviceAddressInfo*);
using PFN_vkCreateAccelerationStructureKHR=VkResult(EVANS_VK_CALL*)(VkDevice,const VkAccelerationStructureCreateInfoKHR*,const void*,VkAccelerationStructureKHR*);
using PFN_vkDestroyAccelerationStructureKHR=void(EVANS_VK_CALL*)(VkDevice,VkAccelerationStructureKHR,const void*);
using PFN_vkGetAccelerationStructureDeviceAddressKHR=VkDeviceAddress(EVANS_VK_CALL*)(VkDevice,const VkAccelerationStructureDeviceAddressInfoKHR*);
using PFN_vkGetAccelerationStructureBuildSizesKHR=void(EVANS_VK_CALL*)(VkDevice,std::int32_t,const VkAccelerationStructureBuildGeometryInfoKHR*,const std::uint32_t*,VkAccelerationStructureBuildSizesInfoKHR*);
using PFN_vkCmdBuildAccelerationStructuresKHR=void(EVANS_VK_CALL*)(VkCommandBuffer,std::uint32_t,const VkAccelerationStructureBuildGeometryInfoKHR*,const VkAccelerationStructureBuildRangeInfoKHR* const*);
}
