#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Transparency/RefractionContract.h"
#include "Renderer/TechniqueShader.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>

namespace Evans::Renderer
{
    // Capture owns depth; composition samples the unchanged main HDR image and
    // writes a private target. Unbind all views before copying the result back.
    class DX11Refraction
    {
        template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Image
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11ShaderResourceView> View;
        };
        std::array<Image,3> Images;
        ComPtr<ID3D11Texture2D> Depth;
        ComPtr<ID3D11DepthStencilView> DepthTarget;
        ComPtr<ID3D11ShaderResourceView> DepthView;
        ComPtr<ID3D11Buffer> Vertices, Indices, Constants;
        std::array<ComPtr<ID3D11VertexShader>,2> VertexShaders;
        std::array<ComPtr<ID3D11PixelShader>,2> PixelShaders;
        ComPtr<ID3D11InputLayout> Layout;
        std::array<ComPtr<ID3D11RasterizerState>,4> Rasters;
        ComPtr<ID3D11DepthStencilState> DepthWrite, DepthDisabled;
        ComPtr<ID3D11SamplerState> Sampler;
        std::shared_ptr<const RefractionGeometry> Source;
        int Width = 0, Height = 0;
    public:
        bool Prepare(ID3D11Device* device,const RefractionFrame& frame,int width,int height,std::string& error)
        {
            if (!frame.Geometry || frame.Draws.empty()) return true;
            auto failed = [&](HRESULT result,const char* operation) {
                if (SUCCEEDED(result)) return false;
                error = std::string("DX11 refraction ")+operation+" failed: "+std::to_string(result); return true;
            };
            if (Source != frame.Geometry)
            {
                const auto& geometry = *frame.Geometry;
                if (geometry.Vertices.empty() || geometry.Indices.empty() || geometry.Vertices.size() > UINT32_MAX/sizeof(Vertex) ||
                    geometry.Indices.size() > UINT32_MAX/sizeof(std::uint32_t)) { error="Invalid DX11 refraction geometry size"; return false; }
                ComPtr<ID3D11Buffer> vertices,indices;
                D3D11_BUFFER_DESC buffer{}; buffer.Usage=D3D11_USAGE_IMMUTABLE;
                buffer.BindFlags=D3D11_BIND_VERTEX_BUFFER; buffer.ByteWidth=static_cast<UINT>(geometry.Vertices.size()*sizeof(Vertex));
                D3D11_SUBRESOURCE_DATA initial{}; initial.pSysMem=geometry.Vertices.data();
                if (failed(device->CreateBuffer(&buffer,&initial,vertices.GetAddressOf()),"vertices")) return false;
                buffer.BindFlags=D3D11_BIND_INDEX_BUFFER; buffer.ByteWidth=static_cast<UINT>(geometry.Indices.size()*sizeof(std::uint32_t));
                initial.pSysMem=geometry.Indices.data();
                if (failed(device->CreateBuffer(&buffer,&initial,indices.GetAddressOf()),"indices")) return false;
                Vertices=std::move(vertices); Indices=std::move(indices); Source=frame.Geometry;
            }
            if (Width != width || Height != height || !Depth)
            {
                std::array<Image,3> images; D3D11_TEXTURE2D_DESC desc{};
                desc.Width=width; desc.Height=height; desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
                desc.Format=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
                for (auto& image:images)
                    if (failed(device->CreateTexture2D(&desc,nullptr,image.Texture.GetAddressOf()),"color image") ||
                        failed(device->CreateRenderTargetView(image.Texture.Get(),nullptr,image.Target.GetAddressOf()),"color target") ||
                        failed(device->CreateShaderResourceView(image.Texture.Get(),nullptr,image.View.GetAddressOf()),"color view")) return false;
                ComPtr<ID3D11Texture2D> depth; ComPtr<ID3D11DepthStencilView> target; ComPtr<ID3D11ShaderResourceView> view;
                desc.Format=DXGI_FORMAT_R32_TYPELESS; desc.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
                D3D11_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
                D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=DXGI_FORMAT_R32_FLOAT; srv.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; srv.Texture2D.MipLevels=1;
                if (failed(device->CreateTexture2D(&desc,nullptr,depth.GetAddressOf()),"private depth") ||
                    failed(device->CreateDepthStencilView(depth.Get(),&dsv,target.GetAddressOf()),"private depth target") ||
                    failed(device->CreateShaderResourceView(depth.Get(),&srv,view.GetAddressOf()),"private depth view")) return false;
                Images=std::move(images); Depth=std::move(depth); DepthTarget=std::move(target); DepthView=std::move(view); Width=width; Height=height;
            }
            if (VertexShaders[0] && VertexShaders[1] && PixelShaders[0] && PixelShaders[1] && Constants && Layout &&
                DepthWrite && DepthDisabled && Sampler && Rasters[0] && Rasters[3]) return true;
            std::string source; if (!LoadTechniqueShaderSource("DX11","Refraction.hlsl",source,error)) return false;
            for (unsigned stage=0;stage<2;++stage)
            {
                const D3D_SHADER_MACRO definitions[]{{"EVANS_REFRACTION_CAPTURE","1"},{nullptr,nullptr}};
                auto compile=[&](const char* entry,const char* profile,ComPtr<ID3DBlob>& code) {
                    ComPtr<ID3DBlob> diagnostics;
                    const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),"Refraction.hlsl",stage==0 ? definitions : nullptr,nullptr,
                        entry,profile,D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,
                        code.GetAddressOf(),diagnostics.GetAddressOf());
                    if (FAILED(result)) { error=diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),diagnostics->GetBufferSize()) : "DX11 refraction compiler failed"; return false; }
                    return true;
                };
                ComPtr<ID3DBlob> vertex,pixel;
                if (!compile("VSMain","vs_5_0",vertex) || !compile("PSMain","ps_5_0",pixel)) return false;
                if (failed(device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,VertexShaders[stage].ReleaseAndGetAddressOf()),"vertex shader") ||
                    failed(device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,PixelShaders[stage].ReleaseAndGetAddressOf()),"pixel shader")) return false;
                if (stage==0)
                {
                    const D3D11_INPUT_ELEMENT_DESC attributes[]{
                        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Normal),D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D11_INPUT_PER_VERTEX_DATA,0}};
                    if (failed(device->CreateInputLayout(attributes,UINT(std::size(attributes)),vertex->GetBufferPointer(),vertex->GetBufferSize(),Layout.ReleaseAndGetAddressOf()),"input layout")) return false;
                }
            }
            D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(RefractionConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
            buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer,nullptr,Constants.ReleaseAndGetAddressOf()),"constants")) return false;
            D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE; depth.DepthFunc=D3D11_COMPARISON_LESS; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;
            if (failed(device->CreateDepthStencilState(&depth,DepthWrite.ReleaseAndGetAddressOf()),"depth state")) return false;
            depth.DepthEnable=FALSE; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
            if (failed(device->CreateDepthStencilState(&depth,DepthDisabled.ReleaseAndGetAddressOf()),"composition depth state")) return false;
            for (unsigned index=0;index<Rasters.size();++index)
            {
                D3D11_RASTERIZER_DESC raster{}; raster.FillMode=D3D11_FILL_SOLID; raster.DepthClipEnable=TRUE;
                raster.CullMode=(index&1) ? D3D11_CULL_BACK : D3D11_CULL_NONE; raster.FrontCounterClockwise=(index&2) ? FALSE : TRUE;
                if (failed(device->CreateRasterizerState(&raster,Rasters[index].ReleaseAndGetAddressOf()),"raster state")) return false;
            }
            D3D11_SAMPLER_DESC sampler{}; sampler.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D11_TEXTURE_ADDRESS_WRAP; sampler.MaxLOD=D3D11_FLOAT32_MAX;
            return !failed(device->CreateSamplerState(&sampler,Sampler.ReleaseAndGetAddressOf()),"sampler");
        }

        template<class TextureLookup>
        bool Resolve(ID3D11DeviceContext* context,const RefractionFrame& frame,const RefractionConstants& base,
            ID3D11Texture2D* scene,ID3D11ShaderResourceView* sceneView,ID3D11ShaderResourceView* opaqueDepth,
            TextureLookup&& texture,std::string& error)
        {
            if (!frame.Geometry || frame.Draws.empty()) return true;
            if (Source!=frame.Geometry || !scene || !sceneView || !opaqueDepth) { error="DX11 refraction inputs changed during a frame"; return false; }
            ID3D11ShaderResourceView* empty[8]{};
            auto unbind=[&] { context->OMSetRenderTargets(0,nullptr,nullptr); context->PSSetShaderResources(0,8,empty); };
            auto upload=[&](const RefractionConstants& value) {
                D3D11_MAPPED_SUBRESOURCE mapped{};
                if (FAILED(context->Map(Constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) { error="DX11 refraction constant mapping failed"; return false; }
                std::memcpy(mapped.pData,&value,sizeof(value)); context->Unmap(Constants.Get(),0); return true;
            };
            unbind(); context->OMSetBlendState(nullptr,nullptr,UINT32_MAX); context->OMSetDepthStencilState(DepthWrite.Get(),0);
            const float clear[4]{}; for (unsigned index=0;index<2;++index) context->ClearRenderTargetView(Images[index].Target.Get(),clear);
            context->ClearDepthStencilView(DepthTarget.Get(),D3D11_CLEAR_DEPTH,1,0);
            ID3D11RenderTargetView* targets[]{Images[0].Target.Get(),Images[1].Target.Get()}; context->OMSetRenderTargets(2,targets,DepthTarget.Get());
            const D3D11_VIEWPORT viewport{0,0,float(Width),float(Height),0,1}; context->RSSetViewports(1,&viewport);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); context->IASetInputLayout(Layout.Get());
            auto* vertices=Vertices.Get(); const UINT stride=sizeof(Vertex),offset=0; context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
            context->IASetIndexBuffer(Indices.Get(),DXGI_FORMAT_R32_UINT,0);
            context->VSSetShader(VertexShaders[0].Get(),nullptr,0); context->PSSetShader(PixelShaders[0].Get(),nullptr,0);
            auto* buffer=Constants.Get(); context->VSSetConstantBuffers(0,1,&buffer); context->PSSetConstantBuffers(0,1,&buffer);
            auto* sampler=Sampler.Get(); context->PSSetSamplers(0,1,&sampler);
            for (const auto& draw:frame.Draws)
            {
                RefractionConstants value; RefractionGeometryRange range;
                if (!BuildRefractionDrawConstants(frame,draw,base,value,range)) { unbind(); error="DX11 refraction draw is invalid"; return false; }
                if (!range.IndexCount) continue;
                auto* diffuse=texture(draw.DiffuseTexture); auto* normal=texture(draw.NormalTexture); auto* opacity=texture(draw.OpacityTexture);
                if ((draw.DiffuseTexture && !diffuse) || (draw.NormalTexture && !normal) || (draw.OpacityTexture && !opacity) || !upload(value))
                { unbind(); if (error.empty()) error="DX11 refraction material texture is invalid"; return false; }
                ID3D11ShaderResourceView* inputs[]{opaqueDepth,diffuse,normal,opacity}; context->PSSetShaderResources(0,4,inputs);
                context->RSSetState(Rasters[(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)].Get());
                context->DrawIndexed(range.IndexCount,range.FirstIndex,0);
            }
            unbind(); auto* target=Images[2].Target.Get(); context->OMSetRenderTargets(1,&target,nullptr);
            context->OMSetDepthStencilState(DepthDisabled.Get(),0); context->RSSetState(Rasters[0].Get()); context->IASetInputLayout(nullptr);
            context->VSSetShader(VertexShaders[1].Get(),nullptr,0); context->PSSetShader(PixelShaders[1].Get(),nullptr,0);
            ID3D11ShaderResourceView* inputs[]{opaqueDepth,nullptr,nullptr,nullptr,sceneView,DepthView.Get(),Images[0].View.Get(),Images[1].View.Get()};
            if (!upload(base)) { unbind(); return false; }
            context->PSSetShaderResources(0,8,inputs); context->Draw(3,0); unbind(); context->CopyResource(scene,Images[2].Texture.Get()); return true;
        }
    };
}
#endif
