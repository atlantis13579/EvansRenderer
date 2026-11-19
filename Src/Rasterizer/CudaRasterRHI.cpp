#include "Rasterizer/CudaRasterRHI.h"

#include "BackendCommon.hpp"
#include "CudaRasterPtx.h"
#include "Rasterizer/D3D11GpuOutput.h"
#include "GpuGeometryFrame.h"
#include "GpuRasterCommon.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <dxgi.h>
#endif

namespace Evans::Rasterizer
{
namespace
{

using GpuTriangle = Detail::GpuRasterTriangle;
using GpuTexture = Detail::GpuRasterTexture;

[[nodiscard]] const std::string& EmbeddedCudaRasterPtx()
{
    static const std::string ptx = []
    {
        std::string combined;
        for (const char* chunk : Detail::CudaRasterPtxChunks) combined += chunk;
        return combined;
    }();
    return ptx;
}

#if defined(_WIN32)

using CuResult = int;
using CuDevice = int;
using CuContext = void*;
using CuModule = void*;
using CuFunction = void*;
using CuDevicePointer = unsigned long long;
constexpr CuResult CuSuccess = 0;

class CudaRuntime
{
public:
    ~CudaRuntime() { Shutdown(); }

    [[nodiscard]] bool Initialize(std::string& error)
    {
        Shutdown();
        DriverLibrary_ = LoadLibraryA("nvcuda.dll");
        if (DriverLibrary_ == nullptr)
        {
            error = "CUDA Driver API library nvcuda.dll is unavailable";
            Shutdown();
            return false;
        }
        if (!LoadFunctions(error))
        {
            Shutdown();
            return false;
        }
        int deviceCount = 0;
        if (!Check(CuInit_(0), "cuInit", error) ||
            !Check(CuDeviceGetCount_(&deviceCount), "cuDeviceGetCount", error) ||
            deviceCount <= 0)
        {
            if (error.empty()) error = "No CUDA-capable device is available";
            Shutdown();
            return false;
        }
        if (!Check(CuDeviceGet_(&Device_, 0), "cuDeviceGet", error) ||
            !Check(CuCtxCreate_(&Context_, 0, Device_), "cuCtxCreate", error))
        {
            Shutdown();
            return false;
        }

        if (!Check(
                CuModuleLoadData_(&Module_, EmbeddedCudaRasterPtx().c_str()),
                "cuModuleLoadData(embedded PTX)",
                error) ||
            !Check(CuModuleGetFunction_(&Kernel_, Module_, "RasterKernel"), "cuModuleGetFunction(RasterKernel)", error) ||
            !Check(CuModuleGetFunction_(&TransformKernel_,Module_,"TransformKernel"),"Load CUDA transform kernel",error) ||
            !Check(CuModuleGetFunction_(&SetupKernel_,Module_,"SetupKernel"),"Load CUDA setup kernel",error) ||
            !Check(CuModuleGetFunction_(&OutlineKernel_,Module_,"OutlineKernel"),"Load CUDA outline kernel",error) ||
            !Check(CuModuleGetFunction_(&ShadowKernel_, Module_, "ShadowDepthKernel"), "cuModuleGetFunction(ShadowDepthKernel)", error))
        {
            Shutdown();
            return false;
        }
        (void)CuCtxSetCurrent_(nullptr);
        return true;
    }

    void Shutdown() noexcept
    {
        if (Context_ != nullptr && CuCtxSetCurrent_ != nullptr)
        {
            (void)CuCtxSetCurrent_(Context_);
        }
        ReleaseInterop();
        for(auto buffer:{&Source_,&Transformed_,&Projected_,&Bins_,&Constants_,&Shadow_,&Colors_,&Depths_,&Visibility_}) {
            if(buffer->Pointer && CuMemFree_)(void)CuMemFree_(buffer->Pointer); *buffer={};
        }
        ResidentScene_.reset(); GpuOutput_.Shutdown();
        if(TextureDescriptors_ && CuMemFree_)(void)CuMemFree_(TextureDescriptors_);
        if(TextureTexels_ && CuMemFree_)(void)CuMemFree_(TextureTexels_);
        TextureDescriptors_=TextureTexels_=0;TextureGeneration_=0;TextureCache_.Reset();
        if (Module_ != nullptr && CuModuleUnload_ != nullptr) (void)CuModuleUnload_(Module_);
        if (Context_ != nullptr && CuCtxDestroy_ != nullptr) (void)CuCtxDestroy_(Context_);
        Module_ = nullptr;
        Kernel_ = nullptr;
        ShadowKernel_ = nullptr;
        Context_ = nullptr;
        Device_ = 0;
        if (DriverLibrary_ != nullptr) FreeLibrary(DriverLibrary_);
        DriverLibrary_ = nullptr;
        ResetFunctions();
    }

    [[nodiscard]] RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& externalCancellation,
        const std::atomic<bool>& backendCancellation,
        RasterStatistics& statistics)
    {
        const auto cancelled = [&] { return Detail::IsGpuRasterCancelled(externalCancellation,backendCancellation); };
        if (cancelled()) return RasterStatus::Cancelled;
        const auto start = std::chrono::steady_clock::now();
        const Detail::GpuGeometryFrame frame(scene,settings,target);
        Require(CuCtxSetCurrent_(Context_),"Set CUDA context");
        struct CurrentContext { CudaRuntime& Runtime; ~CurrentContext() { (void)Runtime.CuCtxSetCurrent_(nullptr); } } current{*this};
        const auto bytes = std::size_t(frame.TriangleCount)*sizeof(GpuTriangle);
        auto source = Reserve(Source_,bytes);
        const bool upload = settings.ImmutableScene.get()!=&scene || ResidentScene_.lock().get()!=&scene;
        if (upload)
        {
            ResidentScene_.reset();
            const auto packed=Detail::BuildGpuRasterTriangles(scene);
            if (bytes) Require(CuMemcpyHtoD_(source,packed.data(),bytes),"Upload CUDA source geometry");
            ResidentScene_=settings.ImmutableScene.get()==&scene ? settings.ImmutableScene : nullptr;
        }
        std::string error;
        if (!EnsureGpuTextures(TextureCache_.Get(scene),error)) throw std::runtime_error(error);
        if (cancelled()) return RasterStatus::Cancelled;
        auto triangles=Reserve(Transformed_,bytes);
        auto projected=Reserve(Projected_,std::size_t(frame.TriangleCount)*9*32);
        auto bins=Reserve(Bins_,4*std::max(frame.MainBins.StorageWords(),frame.GenerateShadow?frame.ShadowBins.StorageWords():0));
        auto constants=Reserve(Constants_,sizeof(frame.Constants));
        auto shadow=Reserve(Shadow_,std::size_t(frame.ShadowWidth)*frame.ShadowHeight*sizeof(float));
        Require(CuMemcpyHtoD_(constants,&frame.Constants,sizeof(frame.Constants)),"Upload CUDA frame constants");
        if (frame.ExternalShadow) Require(CuMemcpyHtoD_(shadow,settings.SceneShadowMap->Depth.data(),settings.SceneShadowMap->Depth.size()*sizeof(float)),"Upload external shadow map");
        const auto rasterStart=std::chrono::steady_clock::now();
        statistics.TransformMilliseconds=std::chrono::duration<double,std::milli>(rasterStart-start).count();
        const auto launch=[&](CuFunction kernel,unsigned gx,unsigned gy,unsigned bx,unsigned by,void** args) {
            if (gx && gy) Require(CuLaunchKernel_(kernel,gx,gy,1,bx,by,1,0,nullptr,args,nullptr),"Launch CUDA raster stage");
        };
        void* transformArgs[]{&source,&triangles,&constants};
        launch(TransformKernel_,frame.GroupsX,frame.GroupsY,64,1,transformArgs);
        auto setup=[&](unsigned light,const Detail::GpuBinLayout& layout) {
            Require(CuMemsetD32_(bins,0,layout.StorageWords()),"Clear CUDA tile masks");
            void* args[]{&triangles,&projected,&bins,&constants,&light};
            launch(SetupKernel_,frame.GroupsX,frame.GroupsY,64,1,args);
        };
        if (frame.GenerateShadow)
        {
            setup(1,frame.ShadowBins);
            void* args[]{&triangles,&constants,&bins,&projected,&shadow,&TextureDescriptors_,&TextureTexels_};
            launch(ShadowKernel_,(frame.ShadowWidth+7)/8,(frame.ShadowHeight+7)/8,8,8,args);
        }
        if (cancelled()) return RasterStatus::Cancelled;
        setup(0,frame.MainBins);
        const auto pixels=std::size_t(target.Width())*target.Height();
        CuDevicePointer colors=0,depths=0;
        CuDevicePointer visibility=settings.OutlineEnabled?Reserve(Visibility_,pixels*sizeof(float)):0;
        bool mapped=false;
        try
        {
            if(settings.GpuPresentation)
            {
                EnsureInterop(pixels);
                Require(CuGraphicsMapResources_(1,&Interop_,nullptr),"Map CUDA output buffers");
                mapped=true;
                std::size_t colorBytes=0;
                Require(CuGraphicsResourceGetMappedPointer_(&colors,&colorBytes,Interop_),"Get mapped CUDA color");
                if(colorBytes<pixels*sizeof(Color)) throw std::runtime_error("CUDA output buffer is too small");
            }
            else {colors=Reserve(Colors_,pixels*sizeof(Color));depths=Reserve(Depths_,pixels*sizeof(float));}
            void* args[]{&triangles,&constants,&bins,&projected,&shadow,&TextureDescriptors_,&TextureTexels_,&colors,&depths,&visibility};
            launch(Kernel_,(target.Width()+7)/8,(target.Height()+7)/8,8,8,args);
            if (settings.OutlineEnabled)
            {
                void* outlineArgs[]{&constants,&visibility,&colors};
                launch(OutlineKernel_,(target.Width()+7)/8,(target.Height()+7)/8,8,8,outlineArgs);
            }
            if(mapped)
            {
                Require(CuGraphicsUnmapResources_(1,&Interop_,nullptr),"Unmap CUDA output buffers"); mapped=false;
                target.GpuColor=GpuOutput_.Publish(InteropBuffer_.Get(),target.Width(),target.Height());
            }
            else
            {
                Require(CuMemcpyDtoH_(target.MutableColors().data(),colors,pixels*sizeof(Color)),"Capture CUDA color");
                Require(CuMemcpyDtoH_(target.MutableDepths().data(),depths,pixels*sizeof(float)),"Capture CUDA depth");
                statistics.ShadedFragments=std::count_if(target.Depths().begin(),target.Depths().end(),[](float z){return z>=0 && z<=1;});
            }
        }
        catch (...) { if(mapped)(void)CuGraphicsUnmapResources_(1,&Interop_,nullptr);throw; }
        statistics.PostClipCountAvailable=frame.Constants.CollectCounts!=0;
        if(statistics.PostClipCountAvailable) {
            unsigned count=0;Require(CuMemcpyDtoH_(&count,bins+frame.MainBins.Words*4,sizeof(count)),"Capture CUDA primitive count");
            statistics.PostClipTriangles=count;
        }
        frame.Report(statistics,upload,"Native CUDA");
        statistics.RasterMilliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-rasterStart).count();
        return cancelled()?RasterStatus::Cancelled:RasterStatus::Completed;
    }

private:
    struct Buffer { CuDevicePointer Pointer = 0; std::size_t Capacity = 0; };
    Buffer Source_, Transformed_, Projected_, Bins_, Constants_, Shadow_, Colors_, Depths_, Visibility_;
    std::weak_ptr<const RasterScene> ResidentScene_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> InteropBuffer_;
    void* Interop_=nullptr;
    std::size_t InteropPixels_=0;
    static void Require(CuResult result,const char* operation)
    {
        std::string error; if (!Check(result,operation,error)) throw std::runtime_error(error);
    }
    CuDevicePointer Reserve(Buffer& buffer,std::size_t bytes)
    {
        bytes=std::max<std::size_t>(bytes,4);
        if(buffer.Capacity<bytes)
        {
            CuDevicePointer replacement=0; Require(CuMemAlloc_(&replacement,bytes),"Allocate CUDA raster buffer");
            if(buffer.Pointer)(void)CuMemFree_(buffer.Pointer);buffer={replacement,bytes};
        }
        return buffer.Pointer;
    }
    void ReleaseInterop() noexcept
    {
        if (Interop_) (void)CuGraphicsUnregisterResource_(Interop_);
        Interop_=nullptr; InteropBuffer_.Reset();
        InteropPixels_=0;
    }
    void EnsureInterop(std::size_t pixels)
    {
        if(InteropPixels_>=pixels)return;
        if(pixels>UINT32_MAX/sizeof(Color))throw std::length_error("CUDA viewport exceeds interop buffer limits");
        EnsureGpuOutput();ReleaseInterop();
        auto buffer=GpuOutput_.CreateInput(static_cast<unsigned>(pixels),sizeof(Color));
        Require(CuGraphicsD3D11RegisterResource_(&Interop_,buffer.Get(),0),"Register CUDA color output");
        InteropBuffer_=std::move(buffer); InteropPixels_=pixels;
    }
    bool EnsureGpuTextures(const Detail::GpuRasterTextureData& data,std::string& error)
    {
        if(TextureGeneration_==TextureCache_.Generation())return true;
        CuDevicePointer descriptors=0,texels=0;
        const auto descriptorBytes=std::max<std::size_t>(data.Descriptors.size()*sizeof(GpuTexture),sizeof(GpuTexture));
        const auto texelBytes=std::max<std::size_t>(data.TexelsRGBA8.size()*sizeof(std::uint32_t),sizeof(std::uint32_t));
        const bool ready=Check(CuMemAlloc_(&descriptors,descriptorBytes),"Allocate CUDA texture descriptors",error) &&
            Check(CuMemAlloc_(&texels,texelBytes),"Allocate CUDA texture texels",error) &&
            (data.Descriptors.empty() || Check(CuMemcpyHtoD_(descriptors,data.Descriptors.data(),data.Descriptors.size()*sizeof(GpuTexture)),"Upload CUDA texture descriptors",error)) &&
            (data.TexelsRGBA8.empty() || Check(CuMemcpyHtoD_(texels,data.TexelsRGBA8.data(),data.TexelsRGBA8.size()*sizeof(std::uint32_t)),"Upload CUDA texture texels",error));
        if(!ready){if(descriptors)(void)CuMemFree_(descriptors);if(texels)(void)CuMemFree_(texels);return false;}
        if(TextureDescriptors_)(void)CuMemFree_(TextureDescriptors_);if(TextureTexels_)(void)CuMemFree_(TextureTexels_);
        TextureDescriptors_=descriptors;TextureTexels_=texels;TextureGeneration_=TextureCache_.Generation();return true;
    }
    void EnsureGpuOutput()
    {
        if(GpuOutput_.Device())return;
        if (!(
            LoadDriver(CuD3D11GetDevice_, "cuD3D11GetDevice") &&
            LoadDriver(CuGraphicsD3D11RegisterResource_, "cuGraphicsD3D11RegisterResource") &&
            LoadDriver(CuGraphicsUnregisterResource_, "cuGraphicsUnregisterResource") &&
            LoadDriver(CuGraphicsMapResources_, "cuGraphicsMapResources") &&
            LoadDriver(CuGraphicsUnmapResources_, "cuGraphicsUnmapResources") &&
            LoadDriver(CuGraphicsResourceGetMappedPointer_, "cuGraphicsResourceGetMappedPointer_v2")))
            throw std::runtime_error("CUDA rendering is available, but CUDA-D3D11 display interop is unavailable");
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))throw std::runtime_error("Create CUDA presentation factory failed");
        for(UINT index=0;;++index)
        {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            if(factory->EnumAdapters1(index,&adapter)==DXGI_ERROR_NOT_FOUND)break;
            CuDevice device=-1;
            if(CuD3D11GetDevice_(&device,adapter.Get())!=CuSuccess || device!=Device_)continue;
            Microsoft::WRL::ComPtr<ID3D11Device> outputDevice;
            D3D_FEATURE_LEVEL level;
            if(FAILED(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,nullptr,0,
                D3D11_SDK_VERSION,&outputDevice,&level,nullptr)))break;
            GpuOutput_.Initialize(outputDevice.Get());return;
        }
        throw std::runtime_error("No D3D11 presentation device matches the CUDA renderer");
    }
    template <typename Function>
    [[nodiscard]] bool LoadDriver(Function& function, const char* name)
    {
        function = reinterpret_cast<Function>(GetProcAddress(DriverLibrary_, name));
        return function != nullptr;
    }

    [[nodiscard]] bool LoadFunctions(std::string& error)
    {
        const bool driverLoaded =
            LoadDriver(CuInit_, "cuInit") &&
            LoadDriver(CuDeviceGetCount_, "cuDeviceGetCount") &&
            LoadDriver(CuDeviceGet_, "cuDeviceGet") &&
            LoadDriver(CuCtxCreate_, "cuCtxCreate_v2") &&
            LoadDriver(CuCtxDestroy_, "cuCtxDestroy_v2") &&
            LoadDriver(CuCtxSetCurrent_, "cuCtxSetCurrent") &&
            LoadDriver(CuModuleLoadData_, "cuModuleLoadData") &&
            LoadDriver(CuModuleUnload_, "cuModuleUnload") &&
            LoadDriver(CuModuleGetFunction_, "cuModuleGetFunction") &&
            LoadDriver(CuMemAlloc_, "cuMemAlloc_v2") &&
            LoadDriver(CuMemFree_, "cuMemFree_v2") &&
            LoadDriver(CuMemcpyHtoD_, "cuMemcpyHtoD_v2") &&
            LoadDriver(CuMemcpyDtoH_, "cuMemcpyDtoH_v2") &&
            LoadDriver(CuLaunchKernel_, "cuLaunchKernel") &&
            LoadDriver(CuMemsetD32_, "cuMemsetD32_v2");
        if (!driverLoaded)
        {
            error = "CUDA Driver API is missing required entry points";
            return false;
        }
        return true;
    }

    [[nodiscard]] static bool Check(CuResult result, const char* operation, std::string& error)
    {
        if (result == CuSuccess) return true;
        error = std::string(operation) + " failed with CUDA error " + std::to_string(result);
        return false;
    }

    void ResetFunctions() noexcept
    {
        CuInit_ = nullptr; CuDeviceGetCount_ = nullptr; CuDeviceGet_ = nullptr;
        CuCtxCreate_ = nullptr; CuCtxDestroy_ = nullptr;
        CuCtxSetCurrent_ = nullptr; CuModuleLoadData_ = nullptr; CuModuleUnload_ = nullptr;
        CuModuleGetFunction_ = nullptr; CuMemAlloc_ = nullptr; CuMemFree_ = nullptr;
        CuMemcpyHtoD_ = nullptr; CuMemcpyDtoH_ = nullptr; CuLaunchKernel_ = nullptr;
        CuMemsetD32_ = nullptr;
    }

    CuDevicePointer TextureDescriptors_=0,TextureTexels_=0;
    std::uint64_t TextureGeneration_=0;
    Detail::GpuRasterTextureCache TextureCache_;
    Detail::D3D11GpuOutput GpuOutput_;
    CuResult (*CuD3D11GetDevice_)(CuDevice*,IDXGIAdapter*) = nullptr;
    CuResult (*CuGraphicsD3D11RegisterResource_)(void**,ID3D11Resource*,unsigned int) = nullptr;
    CuResult (*CuGraphicsUnregisterResource_)(void*) = nullptr;
    CuResult (*CuGraphicsMapResources_)(unsigned int,void**,void*) = nullptr;
    CuResult (*CuGraphicsUnmapResources_)(unsigned int,void**,void*) = nullptr;
    CuResult (*CuGraphicsResourceGetMappedPointer_)(CuDevicePointer*,std::size_t*,void*) = nullptr;
    HMODULE DriverLibrary_ = nullptr;
    CuDevice Device_ = 0;
    CuContext Context_ = nullptr;
    CuModule Module_ = nullptr;
    CuFunction Kernel_ = nullptr;
    CuFunction TransformKernel_=nullptr, SetupKernel_=nullptr;
    CuFunction ShadowKernel_ = nullptr, OutlineKernel_ = nullptr;

    CuResult (*CuInit_)(unsigned int) = nullptr;
    CuResult (*CuDeviceGetCount_)(int*) = nullptr;
    CuResult (*CuDeviceGet_)(CuDevice*, int) = nullptr;
    CuResult (*CuCtxCreate_)(CuContext*, unsigned int, CuDevice) = nullptr;
    CuResult (*CuCtxDestroy_)(CuContext) = nullptr;
    CuResult (*CuCtxSetCurrent_)(CuContext) = nullptr;
    CuResult (*CuModuleLoadData_)(CuModule*, const void*) = nullptr;
    CuResult (*CuModuleUnload_)(CuModule) = nullptr;
    CuResult (*CuModuleGetFunction_)(CuFunction*, CuModule, const char*) = nullptr;
    CuResult (*CuMemAlloc_)(CuDevicePointer*, std::size_t) = nullptr;
    CuResult (*CuMemFree_)(CuDevicePointer) = nullptr;
    CuResult (*CuMemcpyHtoD_)(CuDevicePointer, const void*, std::size_t) = nullptr;
    CuResult (*CuMemcpyDtoH_)(void*, CuDevicePointer, std::size_t) = nullptr;
    CuResult (*CuLaunchKernel_)(CuFunction, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**) = nullptr;
    CuResult (*CuMemsetD32_)(CuDevicePointer,unsigned,std::size_t) = nullptr;

};

#else

class CudaRuntime
{
public:
    [[nodiscard]] bool Initialize(std::string& error)
    {
        error = "CUDA raster backend is only available on Windows in this build";
        return false;
    }
    void Shutdown() noexcept {}
};

#endif

} // namespace

class CudaRasterRHI::Context
{
public:
    Detail::ExecutionGate Execution;
    Detail::StatisticsStore Statistics;
    CudaRuntime Runtime;
    mutable std::mutex StateMutex;
    std::string Error;
    bool NativeAvailable = false;
};

CudaRasterRHI::CudaRasterRHI()
    : Context_(std::make_unique<Context>())
{
    (void)Initialize({});
}

CudaRasterRHI::~CudaRasterRHI()
{
    Shutdown();
}

bool CudaRasterRHI::Initialize(const RasterRHICreateInfo& createInfo)
{
    Shutdown();
    std::string error;
    const bool available = Context_->Runtime.Initialize(error);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        Context_->NativeAvailable = available;
        Context_->Error = std::move(error);
    }
    return available && Context_->Execution.Initialize(createInfo.WorkerThreads);
}

void CudaRasterRHI::Shutdown() noexcept
{
    Context_->Execution.Shutdown();
    Context_->Runtime.Shutdown();
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    Context_->NativeAvailable = false;
}

bool CudaRasterRHI::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return Context_->NativeAvailable && Context_->Execution.IsInitialized();
}

RasterBackend CudaRasterRHI::Backend() const noexcept
{
    return RasterBackend::CUDA;
}

std::string_view CudaRasterRHI::Name() const noexcept
{
    return "CUDA Driver Raster";
}

RasterBackendCapabilities CudaRasterRHI::Capabilities() const
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return {
        RasterBackend::CUDA,
        Context_->NativeAvailable,
        Context_->NativeAvailable,
        8,
        8,
        Context_->NativeAvailable
            ? "Native CUDA Driver API with embedded PTX, tiled raster and shadow kernels"
            : "Unsupported: " + Context_->Error};
}

RasterStatus CudaRasterRHI::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    Detail::ValidateRenderTarget(target);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        if (!Context_->NativeAvailable)
        {
            throw std::runtime_error("CUDA raster backend is unsupported: " + Context_->Error);
        }
    }
    RasterStatistics statistics;
    const RasterStatus status = Context_->Execution.Execute(
        [&](const std::atomic<bool>& backendCancellation)
        {
#if defined(_WIN32)
            return Context_->Runtime.Render(
                scene, settings, target, cancellation, backendCancellation, statistics);
#else
            (void)scene; (void)settings; (void)target; (void)cancellation; (void)backendCancellation;
            return RasterStatus::Cancelled;
#endif
        });
    Context_->Statistics.Set(std::move(statistics));
    return status;
}

void CudaRasterRHI::Cancel() noexcept
{
    Context_->Execution.Cancel();
}

void CudaRasterRHI::WaitIdle() noexcept
{
    Context_->Execution.WaitIdle();
}

RasterStatistics CudaRasterRHI::LastStatistics() const
{
    return Context_->Statistics.Get();
}

} // namespace Evans::Rasterizer
