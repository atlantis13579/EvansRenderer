#include "Renderer/Vulkan/VulkanPipelineCache.h"
#include "Renderer/Vulkan/VulkanDeviceProperties.h"
#include "Renderer/ShaderCache.h"
#include <cstring>
#include <map>
#include <mutex>
namespace Evans::Renderer
{
    using namespace VulkanABI;
    namespace
    {
        struct CacheCreateInfo
        { VkStructureType sType=17;const void* pNext=nullptr;VkFlags flags=0;std::size_t initialDataSize=0;const void* pInitialData=nullptr; };
        using CreateCache=VkResult(EVANS_VK_CALL*)(VkDevice,const CacheCreateInfo*,const void*,VkPipelineCache*);
        using DestroyCache=void(EVANS_VK_CALL*)(VkDevice,VkPipelineCache,const void*);
        using GetCache=VkResult(EVANS_VK_CALL*)(VkDevice,VkPipelineCache,std::size_t*,void*);
        struct State
        { VkPipelineCache Cache=0;DestroyCache Destroy=nullptr;GetCache Get=nullptr;ShaderCache::Digest Key{}; };
        // Vulkan requires host synchronization for pipeline-cache access. All
        // producers sharing a device, including async techniques, use this lock.
        std::mutex Mutex;std::map<VkDevice,State> States;
    }
    void InitializeVulkanPipelineCache(VkDevice device,VkPhysicalDevice physical,
        PFN_vkGetPhysicalDeviceProperties properties,PFN_vkGetDeviceProcAddr address)
    {
        if(!ShaderCache::Enabled() || !device || !properties || !address)return;
        std::lock_guard<std::mutex> lock(Mutex);
        if(States.count(device))return;
        const auto create=reinterpret_cast<CreateCache>(address(device,"vkCreatePipelineCache"));
        State state;state.Destroy=reinterpret_cast<DestroyCache>(address(device,"vkDestroyPipelineCache"));
        state.Get=reinterpret_cast<GetCache>(address(device,"vkGetPipelineCacheData"));
        if(!create || !state.Destroy || !state.Get)return;
        alignas(8) std::array<std::uint8_t,4096> storage{};properties(physical,storage.data());
        PhysicalDevicePropertiesPrefix decoded{};std::memcpy(&decoded,storage.data(),sizeof(decoded));
        ShaderCache::Key key("vulkan-pipeline-v1");key.Value(decoded.VendorId).Value(decoded.DeviceId).Value(decoded.DriverVersion)
            .Value(decoded.ApiVersion).Add(decoded.PipelineCacheUuid,sizeof(decoded.PipelineCacheUuid));state.Key=key.Finish();
        std::vector<std::uint8_t> data;const bool cached=ShaderCache::Read("vulkan-pipeline",state.Key,data);
        CacheCreateInfo info;info.initialDataSize=data.size();info.pInitialData=data.empty()?nullptr:data.data();
        auto result=create(device,&info,nullptr,&state.Cache);
        if(result!=VK_SUCCESS && cached)
        { info.initialDataSize=0;info.pInitialData=nullptr;result=create(device,&info,nullptr,&state.Cache); }
        if(result==VK_SUCCESS)States.emplace(device,state);
    }
    void DestroyVulkanPipelineCache(VkDevice device)
    {
        std::lock_guard<std::mutex> lock(Mutex);const auto found=States.find(device);if(found==States.end())return;
        const auto& state=found->second;std::size_t size=0;
        if(state.Get(device,state.Cache,&size,nullptr)==VK_SUCCESS && size && size<=256*1024*1024)
        {
            std::vector<std::uint8_t> data(size);
            if(state.Get(device,state.Cache,&size,data.data())==VK_SUCCESS)
                ShaderCache::Write("vulkan-pipeline",state.Key,data.data(),size);
        }
        state.Destroy(device,state.Cache,nullptr);States.erase(found);
    }
    VkResult CreateCachedVulkanGraphicsPipelines(PFN_vkCreateGraphicsPipelines create,VkDevice device,VkPipelineCache cache,
        std::uint32_t count,const VkGraphicsPipelineCreateInfo* descriptions,const void* allocator,VkPipeline* pipelines)
    {
        std::lock_guard<std::mutex> lock(Mutex);const auto found=States.find(device);
        if(!cache && found!=States.end())cache=found->second.Cache;
        return create(device,cache,count,descriptions,allocator,pipelines);
    }
    VkResult CreateCachedVulkanComputePipelines(PFN_vkCreateComputePipelines create,VkDevice device,VkPipelineCache cache,
        std::uint32_t count,const VkComputePipelineCreateInfo* descriptions,const void* allocator,VkPipeline* pipelines)
    {
        std::lock_guard<std::mutex> lock(Mutex);const auto found=States.find(device);
        if(!cache && found!=States.end())cache=found->second.Cache;
        return create(device,cache,count,descriptions,allocator,pipelines);
    }
}
