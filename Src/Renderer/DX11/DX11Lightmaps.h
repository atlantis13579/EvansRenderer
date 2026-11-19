#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/BakedLighting/LightmapContract.h"
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
    class DX11Lightmaps
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Surface
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
    public:
        bool Prepare(ID3D11Device* device, const LightmapFrame& frame, int width, int height, UINT samples, std::string& error)
        {
            if (!frame.Source || frame.Draws.empty()) return true;
            if (!frame.Source->Irradiance || !frame.Source->Receivers || !frame.Source->Irradiance->Atlas)
            { error = "DX11 lightmaps require a complete receiver and bake snapshot"; return false; }
            auto failed = [&](HRESULT result,const char* operation) {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 lightmaps ")+operation+" failed: "+std::to_string(result); return true;
            };
            auto allocate = [&](Surface& output,UINT w,UINT h,UINT sampleCount,const Float4* data) {
                Surface candidate; D3D11_TEXTURE2D_DESC desc{};
                desc.Width=w; desc.Height=h; desc.MipLevels=desc.ArraySize=1; desc.SampleDesc.Count=sampleCount;
                desc.Format=data ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
                desc.Usage=data ? D3D11_USAGE_IMMUTABLE : D3D11_USAGE_DEFAULT;
                desc.BindFlags=D3D11_BIND_SHADER_RESOURCE | (data ? 0 : D3D11_BIND_RENDER_TARGET);
                D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=data; initial.SysMemPitch=w*sizeof(Float4);
                if (failed(device->CreateTexture2D(&desc,data ? &initial : nullptr,candidate.Texture.GetAddressOf()),"texture") ||
                    failed(device->CreateShaderResourceView(candidate.Texture.Get(),nullptr,candidate.View.GetAddressOf()),"texture view") ||
                    (!data && failed(device->CreateRenderTargetView(candidate.Texture.Get(),nullptr,candidate.Target.GetAddressOf()),"target"))) return false;
                output=std::move(candidate); return true;
            };
            if (Source != frame.Source)
            {
                const auto& bake=*frame.Source->Irradiance; const auto side=bake.Atlas->Settings.Resolution;
                const auto& vertices=frame.Source->Receivers->Vertices;
                if (vertices.empty() || vertices.size()>UINT_MAX/sizeof(LightmapReceiverVertex) ||
                    bake.FrontIrradiance.size()!=static_cast<std::size_t>(side)*side ||
                    (!bake.BackIrradiance.empty() && bake.BackIrradiance.size()!=bake.FrontIrradiance.size()))
                { error="DX11 lightmap upload size is invalid"; return false; }
                Surface front,back; ComPtr<ID3D11Buffer> vertexBuffer;
                if (!allocate(front,side,side,1,bake.FrontIrradiance.data()) ||
                    !allocate(back,side,side,1,bake.BackIrradiance.empty() ? bake.FrontIrradiance.data() : bake.BackIrradiance.data())) return false;
                D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(vertices.size()*sizeof(LightmapReceiverVertex));
                desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
                D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=vertices.data();
                if (failed(device->CreateBuffer(&desc,&initial,vertexBuffer.GetAddressOf()),"receiver buffer")) return false;
                Front=std::move(front); Back=std::move(back); Vertices=std::move(vertexBuffer); Source=frame.Source;
            }
            if (width!=Width || height!=Height || samples!=Samples || !Composite.Texture)
            {
                if (!allocate(Covered,width,height,samples,nullptr) || !allocate(Resolved,width,height,1,nullptr) ||
                    !allocate(Composite,width,height,1,nullptr)) return false;
                Width=width; Height=height; Samples=samples;
            }
            if (Vertex && Pixel && Constants && Layout && DepthEqual && DepthDisabled && FirstSurface &&  Rasters[0] && Rasters[3]) return true;
            std::string source; if (!LoadTechniqueShaderSource("DX11","Lightmaps.hlsl",source,error)) return false;
            auto compile = [&](const char* entry,const char* profile,ComPtr<ID3DBlob>& bytecode) {
                ComPtr<ID3DBlob> diagnostics;
                const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),"Lightmaps.hlsl",nullptr,nullptr,entry,profile,
                    D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS|D3DCOMPILE_OPTIMIZATION_LEVEL3,0,
                    bytecode.GetAddressOf(),diagnostics.GetAddressOf());
                if (FAILED(result)) { error=diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),diagnostics->GetBufferSize()) : "DX11 lightmap shader compilation failed"; return false; }
                return true;
            };
            ComPtr<ID3DBlob> vertex,pixel;
            if (!compile("VSMain","vs_5_0",vertex) || !compile("PSMain","ps_5_0",pixel)) return false;
            const D3D11_INPUT_ELEMENT_DESC attributes[] {
                {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",1,DXGI_FORMAT_R32G32_FLOAT,0,20,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,28,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if (failed(device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,Vertex.ReleaseAndGetAddressOf()),"vertex shader") ||
                failed(device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,Pixel.ReleaseAndGetAddressOf()),"pixel shader") ||
                failed(device->CreateInputLayout(attributes,static_cast<UINT>(std::size(attributes)),vertex->GetBufferPointer(),vertex->GetBufferSize(),Layout.ReleaseAndGetAddressOf()),"receiver layout")) return false;
            D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(LightmapConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
            buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer,nullptr,Constants.ReleaseAndGetAddressOf()),"uniform buffer")) return false;
            D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE; depth.DepthFunc=D3D11_COMPARISON_EQUAL; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
            if (failed(device->CreateDepthStencilState(&depth,DepthEqual.ReleaseAndGetAddressOf()),"read-only depth")) return false;
            depth.DepthEnable=FALSE;
            if (failed(device->CreateDepthStencilState(&depth,DepthDisabled.ReleaseAndGetAddressOf()),"composition depth")) return false;
            D3D11_BLEND_DESC blend{}; auto& attachment=blend.RenderTarget[0]; attachment.BlendEnable=TRUE;
            attachment.SrcBlend=D3D11_BLEND_INV_DEST_ALPHA; attachment.DestBlend=D3D11_BLEND_ONE; attachment.BlendOp=D3D11_BLEND_OP_ADD;
            attachment.SrcBlendAlpha=D3D11_BLEND_ONE; attachment.DestBlendAlpha=D3D11_BLEND_ZERO; attachment.BlendOpAlpha=D3D11_BLEND_OP_ADD;
            attachment.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
            if (failed(device->CreateBlendState(&blend,FirstSurface.ReleaseAndGetAddressOf()),"first surface coverage latch")) return false;
            for (UINT index=0;index<Rasters.size();++index)
            {
                D3D11_RASTERIZER_DESC raster{}; raster.FillMode=D3D11_FILL_SOLID; raster.CullMode=index&1 ? D3D11_CULL_BACK : D3D11_CULL_NONE;
                raster.FrontCounterClockwise=index&2 ? FALSE : TRUE; raster.DepthClipEnable=TRUE; raster.MultisampleEnable=TRUE;
                if (failed(device->CreateRasterizerState(&raster,Rasters[index].ReleaseAndGetAddressOf()),"receiver raster")) return false;
            }
            return true;
        }

        template<class TextureLookup>
        bool Resolve(ID3D11DeviceContext* context,const LightmapFrame& frame,const GlobalIlluminationDesc& desc,
            const Techniques::Settings& settings,ID3D11Texture2D* scene,ID3D11ShaderResourceView* sceneView,
            ID3D11DepthStencilView* opaqueDepth,ID3D11ShaderResourceView* aoVisibility,ID3D11SamplerState* sampler,TextureLookup&& texture,std::uint32_t& passCount,std::string& error)
        {
            passCount=0;
            if (!frame.Source || frame.Draws.empty()) return true;
            if (Source!=frame.Source || !opaqueDepth || !scene || !sceneView) { error="DX11 lightmap inputs changed during a frame"; return false; }
            constexpr UINT TextureCount=6; ID3D11ShaderResourceView* empty[TextureCount]{};
            context->OMSetRenderTargets(0,nullptr,nullptr); context->PSSetShaderResources(0,TextureCount,empty);
            context->OMSetBlendState(frame.KeepFirstSurface ? FirstSurface.Get() : nullptr,nullptr,UINT_MAX);
            context->OMSetDepthStencilState(DepthEqual.Get(),0);
            context->IASetInputLayout(Layout.Get()); context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(Vertex.Get(),nullptr,0); context->PSSetShader(Pixel.Get(),nullptr,0);
            auto* vertex=Vertices.Get(); const UINT stride=sizeof(LightmapReceiverVertex),offset=0;
            context->IASetVertexBuffers(0,1,&vertex,&stride,&offset);
            auto* constantsBuffer=Constants.Get(); context->VSSetConstantBuffers(0,1,&constantsBuffer); context->PSSetConstantBuffers(0,1,&constantsBuffer);
            context->PSSetSamplers(0,1,&sampler);
            const D3D11_VIEWPORT viewport{0,0,float(Width),float(Height),0,1}; context->RSSetViewports(1,&viewport);
            const float black[4]{}; context->ClearRenderTargetView(Covered.Target.Get(),black);
            auto* target=Covered.Target.Get(); context->OMSetRenderTargets(1,&target,opaqueDepth);
            LightmapConstants constants;
            auto upload = [&] {
                D3D11_MAPPED_SUBRESOURCE mapped{};
                if (FAILED(context->Map(Constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) { error="DX11 lightmap constants mapping failed"; return false; }
                std::memcpy(mapped.pData,&constants,sizeof(constants)); context->Unmap(Constants.Get(),0); return true;
            };
            for (const auto& draw:frame.Draws)
            {
                if (!BuildLightmapConstants(frame,draw,desc.ViewProjection,settings.GlobalIllumination.Intensity,settings.Shading,settings.Metallic,constants))
                { error="DX11 lightmap receiver range or transform is invalid"; return false; }
                if (!upload()) return false;
                const UINT raster=(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u);
                context->RSSetState(Rasters[raster].Get());
                ID3D11ShaderResourceView* inputs[TextureCount]{Front.View.Get(),Back.View.Get(),texture(draw.DiffuseTexture),texture(draw.OpacityTexture),nullptr,nullptr};
                if ((draw.DiffuseTexture && !inputs[2]) || (draw.OpacityTexture && !inputs[3]))
                { error="DX11 lightmap material texture is invalid"; return false; }
                context->PSSetShaderResources(0,TextureCount,inputs); context->Draw(draw.VertexCount,draw.FirstVertex);
            }
            context->OMSetRenderTargets(0,nullptr,nullptr); context->PSSetShaderResources(0,TextureCount,empty);
            if (Samples>1) context->ResolveSubresource(Resolved.Texture.Get(),0,Covered.Texture.Get(),0,DXGI_FORMAT_R16G16B16A16_FLOAT);
            else context->CopyResource(Resolved.Texture.Get(),Covered.Texture.Get());
            constants.Options.X=aoVisibility ? 1u : 0u;
            constants.Options.W=static_cast<UINT>(LightmapPass::Composite); if (!upload()) return false;
            context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
            context->OMSetDepthStencilState(DepthDisabled.Get(),0); context->RSSetState(Rasters[0].Get());
            ID3D11ShaderResourceView* inputs[TextureCount]{aoVisibility,nullptr,nullptr,nullptr,sceneView,Resolved.View.Get()};
            context->PSSetShaderResources(0,TextureCount,inputs); target=Composite.Target.Get(); context->OMSetRenderTargets(1,&target,nullptr);
            context->Draw(3,0); context->OMSetRenderTargets(0,nullptr,nullptr); context->PSSetShaderResources(0,TextureCount,empty);
            context->CopyResource(scene,Composite.Texture.Get()); passCount=2; return true;
        }
    private:
        Surface Front,Back,Covered,Resolved,Composite;
        std::shared_ptr<const LightmapBakeResult> Source;
        ComPtr<ID3D11Buffer> Vertices,Constants;
        ComPtr<ID3D11VertexShader> Vertex;
        ComPtr<ID3D11PixelShader> Pixel;
        ComPtr<ID3D11InputLayout> Layout;
        ComPtr<ID3D11DepthStencilState> DepthEqual,DepthDisabled;
        ComPtr<ID3D11BlendState> FirstSurface;
        std::array<ComPtr<ID3D11RasterizerState>,4> Rasters;
        int Width=0,Height=0; UINT Samples=0;
    };
}
#endif
