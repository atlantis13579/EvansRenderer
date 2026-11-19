#include "Rasterizer/D3D11GpuOutput.h"
#if defined(_WIN32)
#include <d3dcompiler.h>
#include <dxgi.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Evans::Rasterizer::Detail
{
namespace
{
void Check(HRESULT status, const char* operation)
{
    if (FAILED(status)) throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(status));
}
constexpr char EncodeShader[] = R"(
StructuredBuffer<float4> Colors : register(t0);
RWTexture2D<float4> Output : register(u0);
cbuffer Frame : register(b0) { uint Width, Height; };
[numthreads(8,8,1)] void Main(uint3 id : SV_DispatchThreadID)
{
    if(id.x>=Width || id.y>=Height)return;
    uint pixel=id.y*Width+id.x;
    float4 color=Colors[pixel];
    color=saturate(color);
    color.rgb=float3(color.r<=0.0031308?color.r*12.92:1.055*pow(color.r,1.0/2.4)-0.055,
        color.g<=0.0031308?color.g*12.92:1.055*pow(color.g,1.0/2.4)-0.055,
        color.b<=0.0031308?color.b*12.92:1.055*pow(color.b,1.0/2.4)-0.055);
    Output[id.xy]=color;
})";
}
struct D3D11GpuOutput::Surface
{
    Microsoft::WRL::ComPtr<ID3D11Texture2D> Texture;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> View;
    Microsoft::WRL::ComPtr<IDXGIKeyedMutex> Mutex;
    HANDLE Handle=nullptr;
    std::uint32_t Width=0,Height=0;
};
void D3D11GpuOutput::Initialize(ID3D11Device* device)
{
    Shutdown();
    Microsoft::WRL::ComPtr<ID3DBlob> code,errors;
    const auto status=D3DCompile(EncodeShader,sizeof(EncodeShader)-1,nullptr,nullptr,nullptr,"Main","cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
    if(FAILED(status) && errors)throw std::runtime_error(static_cast<const char*>(errors->GetBufferPointer()));
    Check(status,"Compile presentation encoder");
    Check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&Shader_),"Create presentation encoder");
    Device_=device; device->GetImmediateContext(&Context_);
}
void D3D11GpuOutput::Shutdown()
{
    Pool_.clear();Constants_.Reset();
    InputView_.Reset(); Input_.Reset();
    Shader_.Reset();Context_.Reset();Device_.Reset();
}
Microsoft::WRL::ComPtr<ID3D11Buffer> D3D11GpuOutput::CreateInput(std::uint32_t count,std::uint32_t stride)
{
    D3D11_BUFFER_DESC desc{};desc.ByteWidth=count*stride;desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride=stride;
    Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
    Check(Device_->CreateBuffer(&desc,nullptr,&buffer),"Create GPU presentation input");return buffer;
}
std::shared_ptr<Renderer::GpuImage> D3D11GpuOutput::Publish(ID3D11Buffer* colors, std::uint32_t width, std::uint32_t height)
{
    Pool_.erase(std::remove_if(Pool_.begin(),Pool_.end(),[&](const auto& item){return item.use_count()==1 && (item->Width!=width || item->Height!=height);}),Pool_.end());
    std::shared_ptr<Surface> surface;
    for(auto& item:Pool_)if(item.use_count()==1){surface=item;break;}
    if(!surface)
    {
        surface=std::make_shared<Surface>();surface->Width=width;surface->Height=height;
        D3D11_TEXTURE2D_DESC desc{};desc.Width=width;desc.Height=height;desc.MipLevels=1;desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags=D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
        Check(Device_->CreateTexture2D(&desc,nullptr,&surface->Texture),"Create shared raster texture");
        Check(Device_->CreateUnorderedAccessView(surface->Texture.Get(),nullptr,&surface->View),"Create raster output view");
        Check(surface->Texture.As(&surface->Mutex),"Get raster texture mutex");
        Microsoft::WRL::ComPtr<IDXGIResource> resource;Check(surface->Texture.As(&resource),"Get shared raster resource");
        Check(resource->GetSharedHandle(&surface->Handle),"Export raster texture");Pool_.push_back(surface);
    }
    if(Input_.Get()!=colors)
    {
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
        Check(Device_->CreateShaderResourceView(colors,nullptr,&view),"Create presentation color view");
        InputView_=std::move(view);Input_=colors;
    }
    const std::uint32_t constants[4]{width,height,0,0};
    if(!Constants_)
    {
        D3D11_BUFFER_DESC desc{};desc.ByteWidth=sizeof(constants);desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        Check(Device_->CreateBuffer(&desc,nullptr,&Constants_),"Create raster encoder constants");
    }
    Context_->UpdateSubresource(Constants_.Get(),0,nullptr,&constants,0,0);
    const HRESULT acquired=surface->Mutex->AcquireSync(0,5000);
    if(acquired!=S_OK)throw std::runtime_error("Acquire raster output texture failed");
    ID3D11ShaderResourceView* input=InputView_.Get();ID3D11UnorderedAccessView* output=surface->View.Get();
    ID3D11Buffer* cb=Constants_.Get();Context_->CSSetShader(Shader_.Get(),nullptr,0);Context_->CSSetConstantBuffers(0,1,&cb);
    Context_->CSSetShaderResources(0,1,&input);Context_->CSSetUnorderedAccessViews(0,1,&output,nullptr);
    Context_->Dispatch((width+7)/8,(height+7)/8,1);
    input=nullptr;output=nullptr;cb=nullptr;
    Context_->CSSetShaderResources(0,1,&input);Context_->CSSetUnorderedAccessViews(0,1,&output,nullptr);
    Context_->CSSetConstantBuffers(0,1,&cb);Context_->CSSetShader(nullptr,nullptr,0);
    Check(surface->Mutex->ReleaseSync(0),"Release raster output texture");Context_->Flush();
    auto image=std::make_shared<Renderer::GpuImage>();image->Backend=Renderer::RHIBackend::DX11;
    image->SharedHandle=surface->Handle;image->Owner=surface;return image;
}
}
#endif
