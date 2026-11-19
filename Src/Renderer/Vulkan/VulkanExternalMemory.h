#pragma once
#include "Renderer/Vulkan/VulkanABI.h"

namespace Evans::Renderer::VulkanABI
{
// VK_KHR_external_memory_win32 and dedicated image allocation ABI.
constexpr VkFlags ExternalOpaqueWin32 = 0x2;
constexpr std::uint32_t ExternalQueueFamily = 0xfffffffeu;
struct ExternalImageCreateInfo { VkStructureType sType=1000072001; const void* pNext=nullptr; VkFlags handleTypes=ExternalOpaqueWin32; };
struct ExportMemoryInfo { VkStructureType sType=1000072002; const void* pNext=nullptr; VkFlags handleTypes=ExternalOpaqueWin32; };
struct DedicatedImageMemoryInfo { VkStructureType sType=1000127001; const void* pNext=nullptr; VkImage image=0; VkBuffer buffer=0; };
struct ImportWin32MemoryInfo { VkStructureType sType=1000073000; const void* pNext=nullptr; VkFlags handleType=ExternalOpaqueWin32; void* handle=nullptr; const wchar_t* name=nullptr; };
struct GetWin32MemoryInfo { VkStructureType sType=1000073003; const void* pNext=nullptr; VkDeviceMemory memory=0; VkFlags handleType=ExternalOpaqueWin32; };
struct TimelineSemaphoreFeatures { VkStructureType sType=1000207000; const void* pNext=nullptr; VkBool32 timelineSemaphore=1; };
struct TimelineSemaphoreType { VkStructureType sType=1000207002; const void* pNext=nullptr; std::uint32_t semaphoreType=1; std::uint64_t initialValue=0; };
struct TimelineSemaphoreSubmit { VkStructureType sType=1000207003; const void* pNext=nullptr; std::uint32_t waitValueCount=0; const std::uint64_t* waitValues=nullptr; std::uint32_t signalValueCount=0; const std::uint64_t* signalValues=nullptr; };
struct ExportSemaphoreInfo { VkStructureType sType=1000077000; const void* pNext=nullptr; VkFlags handleTypes=ExternalOpaqueWin32; };
struct ExternalSemaphoreCreateInfo { VkStructureType sType=9; const void* pNext=nullptr; VkFlags flags=0; };
struct GetWin32SemaphoreInfo { VkStructureType sType=1000078003; const void* pNext=nullptr; std::uint64_t semaphore=0; VkFlags handleType=ExternalOpaqueWin32; };
struct ImportWin32SemaphoreInfo { VkStructureType sType=1000078000; const void* pNext=nullptr; std::uint64_t semaphore=0; VkFlags flags=0,handleType=ExternalOpaqueWin32; void* handle=nullptr; const wchar_t* name=nullptr; };
using CreateExternalSemaphore = VkResult(EVANS_VK_CALL*)(VkDevice,const ExternalSemaphoreCreateInfo*,const void*,std::uint64_t*);
using DestroyExternalSemaphore = void(EVANS_VK_CALL*)(VkDevice,std::uint64_t,const void*);
using GetSemaphoreWin32Handle = VkResult(EVANS_VK_CALL*)(VkDevice,const GetWin32SemaphoreInfo*,void**);
using ImportSemaphoreWin32Handle = VkResult(EVANS_VK_CALL*)(VkDevice,const ImportWin32SemaphoreInfo*);
using GetMemoryWin32Handle = VkResult(EVANS_VK_CALL*)(VkDevice,const GetWin32MemoryInfo*,void**);
}
