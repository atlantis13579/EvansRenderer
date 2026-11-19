#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/BakedLighting/LightProbeContract.h"
#include "Renderer/TechniqueShader.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>
#include <climits>

namespace Evans::Renderer
{
    class DX11LightProbes : public ProbeProjectionState
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
    public:
        bool Prepare(ID3D11Device* device, const std::shared_ptr<const BakedProbeSamples>& samples,
            int width, int height, std::string& error)
        {
            if (!samples || samples->Texels.empty()) { error = "DX11 light probes require baked ray samples"; return false; }
            auto failed = [&](HRESULT result, const char* operation) {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 light probes ")+operation+" failed: "+std::to_string(result); return true;
            };
            auto allocate = [&](Surface& output, UINT w, UINT h, DXGI_FORMAT format, const Float4* data) {
                Surface result;
                D3D11_TEXTURE2D_DESC desc {}; desc.Width=w; desc.Height=h; desc.MipLevels=desc.ArraySize=1;
                desc.Format=format; desc.SampleDesc.Count=1;
                desc.BindFlags=D3D11_BIND_SHADER_RESOURCE | (data ? 0 : D3D11_BIND_RENDER_TARGET);
                desc.Usage=data ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
                D3D11_SUBRESOURCE_DATA input {}; input.pSysMem=data; input.SysMemPitch=w*sizeof(Float4);
                if (failed(device->CreateTexture2D(&desc,data ? &input : nullptr,result.Texture.GetAddressOf()),"texture") ||
                    failed(device->CreateShaderResourceView(result.Texture.Get(),nullptr,result.View.GetAddressOf()),"view") ||
                    (!data && failed(device->CreateRenderTargetView(result.Texture.Get(),nullptr,result.Target.GetAddressOf()),"target"))) return false;
                output=std::move(result); return true;
            };
            if (samples != Source)
            {
                if (!allocate(Raw,samples->Width(),samples->Height(),DXGI_FORMAT_R32G32B32A32_FLOAT,samples->Texels.data())) return false;
                Invalidate(); Source=samples;
            }
            const auto count=samples->ProbeCount();
            if (count != Count || !Coefficients.Texture || !Moments.Texture)
            {
                if (!allocate(Coefficients,ProbeSHCoefficientCount,count,DXGI_FORMAT_R16G16B16A16_FLOAT,nullptr) ||
                    !allocate(Moments,ProbeMomentAtlasColumns*ProbeMomentTileSize,
                        ((count+ProbeMomentAtlasColumns-1)/ProbeMomentAtlasColumns)*ProbeMomentTileSize,
                        DXGI_FORMAT_R16G16B16A16_FLOAT,nullptr)) return false;
                Count=count; Invalidate();
            }
            if (width != Width || height != Height || !Composite.Texture)
            {
                if (!allocate(Composite,width,height,DXGI_FORMAT_R16G16B16A16_FLOAT,nullptr)) return false;
                Width=width; Height=height;
            }
            if (Vertex && GatherVertex && BakePixel && GatherPixel && Constants && Raster && DepthState) return true;
            auto compile = [&](const char* file, const char* entry, const char* profile, ComPtr<ID3DBlob>& bytecode) {
                std::string source; if (!LoadTechniqueShaderSource("DX11",file,source,error)) return false;
                ComPtr<ID3DBlob> diagnostics;
                const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),file,nullptr,nullptr,entry,profile,
                    D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS|D3DCOMPILE_OPTIMIZATION_LEVEL3,0,
                    bytecode.GetAddressOf(),diagnostics.GetAddressOf());
                if (FAILED(result)) { error=diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),diagnostics->GetBufferSize()) : "DX11 probe shader compilation failed"; return false; }
                return true;
            };
            ComPtr<ID3DBlob> vertex,gatherVertex,bake,gather;
            if (!compile("BakedGI.hlsl","VSMain","vs_5_0",vertex) || !compile("BakedGI.hlsl","PSMain","ps_5_0",bake) ||
                !compile("LightProbes.hlsl","VSMain","vs_5_0",gatherVertex) || !compile("LightProbes.hlsl","PSMain","ps_5_0",gather)) return false;
            if (failed(device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,Vertex.ReleaseAndGetAddressOf()),"vertex shader") ||
                failed(device->CreateVertexShader(gatherVertex->GetBufferPointer(),gatherVertex->GetBufferSize(),nullptr,GatherVertex.ReleaseAndGetAddressOf()),"gather vertex shader") ||
                failed(device->CreatePixelShader(bake->GetBufferPointer(),bake->GetBufferSize(),nullptr,BakePixel.ReleaseAndGetAddressOf()),"bake shader") ||
                failed(device->CreatePixelShader(gather->GetBufferPointer(),gather->GetBufferSize(),nullptr,GatherPixel.ReleaseAndGetAddressOf()),"gather shader")) return false;
            D3D11_BUFFER_DESC buffer {}; buffer.ByteWidth=sizeof(LightProbeConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
            buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            D3D11_RASTERIZER_DESC raster {}; raster.FillMode=D3D11_FILL_SOLID; raster.CullMode=D3D11_CULL_NONE; raster.DepthClipEnable=TRUE;
            D3D11_DEPTH_STENCIL_DESC depth {}; depth.DepthFunc=D3D11_COMPARISON_ALWAYS;
            return !failed(device->CreateBuffer(&buffer,nullptr,Constants.ReleaseAndGetAddressOf()),"constant buffer") &&
                !failed(device->CreateRasterizerState(&raster,Raster.ReleaseAndGetAddressOf()),"rasterizer") &&
                !failed(device->CreateDepthStencilState(&depth,DepthState.ReleaseAndGetAddressOf()),"depth state");
        }

        bool Resolve(ID3D11DeviceContext* context,const GlobalIlluminationDesc& desc,
            const Techniques::GlobalIlluminationSettings& settings,Techniques::ShadingModel shading,
            ID3D11Texture2D* scene,ID3D11ShaderResourceView* color,ID3D11ShaderResourceView* depth,
            ID3D11ShaderResourceView* normal,ID3D11ShaderResourceView* diffuse,ID3D11ShaderResourceView* ambient,
            std::uint32_t& projectionPasses,std::string& error)
        {
            LightProbeConstants constants;
            if (!Source || !BuildLightProbeConstants(desc,*Source,settings,shading,Width,Height,constants))
            { error="DX11 probes require finite geometry and baked samples"; return false; }
            constexpr UINT InputCount=8;
            ID3D11ShaderResourceView* empty[InputCount] {};
            context->OMSetRenderTargets(0,nullptr,nullptr); context->PSSetShaderResources(0,InputCount,empty);
            context->OMSetBlendState(nullptr,nullptr,UINT_MAX); context->OMSetDepthStencilState(DepthState.Get(),0);
            context->RSSetState(Raster.Get()); context->IASetInputLayout(nullptr);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); context->VSSetShader(Vertex.Get(),nullptr,0);
            auto* buffer=Constants.Get(); context->PSSetConstantBuffers(0,1,&buffer);
            auto draw = [&](ProbePass phase,Surface& target,int width,int height,ID3D11ShaderResourceView** inputs) {
                constants.Options.X=static_cast<unsigned>(phase);
                context->VSSetShader(phase==ProbePass::Gather ? GatherVertex.Get() : Vertex.Get(),nullptr,0);
                D3D11_MAPPED_SUBRESOURCE mapping {};
                if (FAILED(context->Map(Constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapping)))
                { error="DX11 probe constants mapping failed"; return false; }
                std::memcpy(mapping.pData,&constants,sizeof(constants)); context->Unmap(Constants.Get(),0);
                D3D11_VIEWPORT viewport {0,0,float(width),float(height),0,1}; context->RSSetViewports(1,&viewport);
                auto* output=target.Target.Get(); context->OMSetRenderTargets(1,&output,nullptr);
                context->PSSetShader(phase==ProbePass::Gather ? GatherPixel.Get() : BakePixel.Get(),nullptr,0);
                context->PSSetShaderResources(0,InputCount,inputs); context->Draw(3,0);
                context->PSSetShaderResources(0,InputCount,empty); context->OMSetRenderTargets(0,nullptr,nullptr); return true;
            };
            projectionPasses=0;
            if (NeedsProjection(Source,constants.Visibility.Y))
            {
                Invalidate();
                ID3D11ShaderResourceView* inputs[InputCount] {Raw.View.Get()};
                if (!draw(ProbePass::ProjectIrradiance,Coefficients,ProbeSHCoefficientCount,Count,inputs) ||
                    !draw(ProbePass::ProjectDistance,Moments,ProbeMomentAtlasColumns*ProbeMomentTileSize,
                        ((Count+ProbeMomentAtlasColumns-1)/ProbeMomentAtlasColumns)*ProbeMomentTileSize,inputs)) return false;
                Projected(Source,constants.Visibility.Y); projectionPasses=2;
            }
            ID3D11ShaderResourceView* inputs[InputCount] {nullptr,Coefficients.View.Get(),Moments.View.Get(),color,depth,normal,diffuse,ambient};
            if (!draw(ProbePass::Gather,Composite,Width,Height,inputs)) return false;
            context->CopyResource(scene,Composite.Texture.Get()); Resolved=true; return true;
        }
    private:
        Surface Raw,Coefficients,Moments,Composite;
        std::shared_ptr<const BakedProbeSamples> Source;
        ComPtr<ID3D11VertexShader> Vertex,GatherVertex;
        ComPtr<ID3D11PixelShader> BakePixel,GatherPixel;
        ComPtr<ID3D11Buffer> Constants;
        ComPtr<ID3D11RasterizerState> Raster;
        ComPtr<ID3D11DepthStencilState> DepthState;
        std::uint32_t Count=0;
        int Width=0,Height=0;
    };
}
#endif
