#include "Renderer/D3DShaderCache.h"
#pragma once
#if defined(_WIN32)
#include "Renderer/Transparency/DepthPeelingContract.h"
#include "Renderer/TechniqueShader.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>

namespace Evans::Renderer
{
    // Private ping-pong depth and transmittance images never alias sampled
    // inputs. The ordinary material pipeline supplies only layer lighting.
    class DX11DepthPeeling
    {
        template<class T> using ComPtr=Microsoft::WRL::ComPtr<T>;
        struct Image
        {
            ComPtr<ID3D11Texture2D> Texture;
            ComPtr<ID3D11ShaderResourceView> View;
            ComPtr<ID3D11RenderTargetView> Target;
            ComPtr<ID3D11DepthStencilView> DepthTarget;
        };
        std::array<Image,2> Depth,Accumulation;
        Image Layer;
        std::array<ComPtr<ID3D11VertexShader>,3> Vertices;
        std::array<ComPtr<ID3D11PixelShader>,3> Pixels;
        ComPtr<ID3D11InputLayout> Layout;
        ComPtr<ID3D11Buffer> Constants;
        std::array<ComPtr<ID3D11RasterizerState>,4> Rasters;
        ComPtr<ID3D11DepthStencilState> WriteDepth,ReadEqual,NoDepth;
        int Width=0,Height=0;
        bool Upload(ID3D11DeviceContext* context,const DepthPeelingConstants& value,std::string& error)
        {
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if (FAILED(context->Map(Constants.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped)))
            { error="DX11 depth-peeling constant upload failed"; return false; }
            std::memcpy(mapped.pData,&value,sizeof(value)); context->Unmap(Constants.Get(),0); return true;
        }
        static void Unbind(ID3D11DeviceContext* context)
        {
            ID3D11ShaderResourceView* empty[8]{};
            context->OMSetRenderTargets(0,nullptr,nullptr); context->PSSetShaderResources(0,8,empty);
        }
        void BindProgram(ID3D11DeviceContext* context,unsigned program)
        {
            context->VSSetShader(Vertices[program].Get(),nullptr,0); context->PSSetShader(Pixels[program].Get(),nullptr,0);
            auto* constants=Constants.Get(); context->VSSetConstantBuffers(0,1,&constants); context->PSSetConstantBuffers(0,1,&constants);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->OMSetBlendState(nullptr,nullptr,UINT32_MAX);
            const D3D11_VIEWPORT viewport{0,0,float(Width),float(Height),0,1}; context->RSSetViewports(1,&viewport);
        }
    public:
        DepthPeelingSequence Sequence;
        ID3D11DepthStencilState* EqualDepthState() const { return ReadEqual.Get(); }
        bool Prepare(ID3D11Device* device,int width,int height,std::string& error)
        {
            auto failed=[&](HRESULT result,const char* operation) {
                if (SUCCEEDED(result)) return false;
                error=std::string("DX11 depth peeling ")+operation+": "+std::to_string(result); return true;
            };
            if (Width!=width || Height!=height || !Layer.Texture)
            {
                std::array<Image,2> depth,accumulation; Image layer;
                auto image=[&](Image& target,DXGI_FORMAT format,bool depthImage) {
                    D3D11_TEXTURE2D_DESC desc{}; desc.Width=width; desc.Height=height;
                    desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1; desc.Format=format;
                    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|(depthImage ? D3D11_BIND_DEPTH_STENCIL : D3D11_BIND_RENDER_TARGET);
                    if (failed(device->CreateTexture2D(&desc,nullptr,target.Texture.GetAddressOf()),"image")) return false;
                    if (depthImage)
                    {
                        D3D11_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format=DXGI_FORMAT_D32_FLOAT; dsv.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
                        D3D11_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=DXGI_FORMAT_R32_FLOAT; srv.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2D; srv.Texture2D.MipLevels=1;
                        return !failed(device->CreateDepthStencilView(target.Texture.Get(),&dsv,target.DepthTarget.GetAddressOf()),"depth target") &&
                            !failed(device->CreateShaderResourceView(target.Texture.Get(),&srv,target.View.GetAddressOf()),"depth view");
                    }
                    return !failed(device->CreateRenderTargetView(target.Texture.Get(),nullptr,target.Target.GetAddressOf()),"color target") &&
                        !failed(device->CreateShaderResourceView(target.Texture.Get(),nullptr,target.View.GetAddressOf()),"color view");
                };
                for (unsigned index=0;index<2;++index)
                    if (!image(depth[index],DXGI_FORMAT_R32_TYPELESS,true) || !image(accumulation[index],DXGI_FORMAT_R32G32B32A32_FLOAT,false)) return false;
                if (!image(layer,DXGI_FORMAT_R16G16B16A16_FLOAT,false)) return false;
                Depth=std::move(depth); Accumulation=std::move(accumulation); Layer=std::move(layer); Width=width; Height=height;
            }
            if (Vertices[2] && Pixels[2] && Constants && Layout &&  Rasters[3] && ReadEqual && NoDepth) return true;
            std::string source; if (!LoadTechniqueShaderSource("DX11","OIT.hlsl",source,error)) return false;
            for (unsigned phase=0;phase<3;++phase)
            {
                const D3D_SHADER_MACRO definitions[]{{phase==0 ? "EVANS_OIT_CAPTURE" : "EVANS_OIT_ACCUMULATE","1"},{nullptr,nullptr}};
                auto compile=[&](const char* entry,const char* profile,ComPtr<ID3DBlob>& code) {
                    ComPtr<ID3DBlob> diagnostics;
                    const auto result=Evans::Renderer::CompileCachedD3DShader(source.data(),source.size(),"OIT.hlsl",phase<2 ? definitions : nullptr,nullptr,entry,profile,
                        D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,code.GetAddressOf(),diagnostics.GetAddressOf());
                    if (FAILED(result)) { error=diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),diagnostics->GetBufferSize()) : "DX11 OIT compilation failed"; return false; }
                    return true;
                };
                ComPtr<ID3DBlob> vertex,pixel;
                if (!compile("VSMain","vs_5_0",vertex) || !compile("PSMain","ps_5_0",pixel) ||
                    failed(device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,Vertices[phase].ReleaseAndGetAddressOf()),"vertex shader") ||
                    failed(device->CreatePixelShader(pixel->GetBufferPointer(),pixel->GetBufferSize(),nullptr,Pixels[phase].ReleaseAndGetAddressOf()),"pixel shader")) return false;
                if (phase==0)
                {
                    const D3D11_INPUT_ELEMENT_DESC attributes[]{
                        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Position),D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(Vertex,Normal),D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(Vertex,Color),D3D11_INPUT_PER_VERTEX_DATA,0},
                        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(Vertex,TexCoord),D3D11_INPUT_PER_VERTEX_DATA,0}};
                    if (failed(device->CreateInputLayout(attributes,UINT(std::size(attributes)),vertex->GetBufferPointer(),vertex->GetBufferSize(),Layout.ReleaseAndGetAddressOf()),"input layout")) return false;
                }
            }
            D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(DepthPeelingConstants); buffer.Usage=D3D11_USAGE_DYNAMIC;
            buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            if (failed(device->CreateBuffer(&buffer,nullptr,Constants.ReleaseAndGetAddressOf()),"constants")) return false;
            D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE; depth.DepthFunc=D3D11_COMPARISON_LESS; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;
            if (failed(device->CreateDepthStencilState(&depth,WriteDepth.ReleaseAndGetAddressOf()),"capture depth state")) return false;
            depth.DepthFunc=D3D11_COMPARISON_EQUAL; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
            if (failed(device->CreateDepthStencilState(&depth,ReadEqual.ReleaseAndGetAddressOf()),"replay depth state")) return false;
            depth.DepthEnable=FALSE;
            if (failed(device->CreateDepthStencilState(&depth,NoDepth.ReleaseAndGetAddressOf()),"fullscreen depth state")) return false;
            for (unsigned index=0;index<Rasters.size();++index)
            {
                D3D11_RASTERIZER_DESC raster{}; raster.FillMode=D3D11_FILL_SOLID; raster.DepthClipEnable=TRUE;
                raster.CullMode=(index&1) ? D3D11_CULL_BACK : D3D11_CULL_NONE; raster.FrontCounterClockwise=(index&2) ? FALSE : TRUE;
                if (failed(device->CreateRasterizerState(&raster,Rasters[index].ReleaseAndGetAddressOf()),"raster state")) return false;
            }
            return true;
        }
        bool BeginLayer(ID3D11DeviceContext* context,ID3D11RenderTargetView*& color,ID3D11DepthStencilView*& depth,std::string& error)
        {
            if (!Sequence.Advance(DepthPeelingEvent::BeginShading,error)) return false;
            Unbind(context); color=Layer.Target.Get(); depth=Depth[Sequence.Step().DepthWrite].DepthTarget.Get();
            const float empty[4]{}; context->ClearRenderTargetView(color,empty); return true;
        }
        template<class BindGeometry,class TextureLookup>
        bool Process(ID3D11DeviceContext* context,DepthPeelingPass pass,const DepthPeelingFrame& frame,ID3D11SamplerState* sampler,
            ID3D11Texture2D* scene,ID3D11ShaderResourceView* sceneView,ID3D11ShaderResourceView* opaqueDepth,
            BindGeometry&& geometry,TextureLookup&& texture,std::string& error)
        {
            const auto step=Sequence.Step(); DepthPeelingConstants base;
            if (!BuildDepthPeelingConstants(frame.ViewProjection,Width,Height,step.FirstLayer,frame.Settings.ShowOverflow,base))
            { error="Invalid DX11 depth-peeling frame constants"; return false; }
            if (pass==DepthPeelingPass::Capture)
            {
                if (!Sequence.Advance(DepthPeelingEvent::BeginCapture,error)) return false;
                Unbind(context); BindProgram(context,0); context->IASetInputLayout(Layout.Get());
                context->OMSetDepthStencilState(WriteDepth.Get(),0);
                if (step.FirstLayer)
                {
                    const float initial[4]{0,0,0,1}; for (auto& image:Accumulation) context->ClearRenderTargetView(image.Target.Get(),initial);
                    for (auto& image:Depth) context->ClearDepthStencilView(image.DepthTarget.Get(),D3D11_CLEAR_DEPTH,PeelingClearDepth,0);
                }
                auto* target=Depth[step.DepthWrite].DepthTarget.Get(); context->ClearDepthStencilView(target,D3D11_CLEAR_DEPTH,PeelingClearDepth,0);
                context->OMSetRenderTargets(0,nullptr,target); context->PSSetSamplers(0,1,&sampler);
                for (const auto& draw:frame.Draws)
                {
                    if (!draw.IndexCount) continue;
                    DepthPeelingConstants value;
                    if (!BuildDepthPeelingDrawConstants(base,draw.World,draw.Opacity,draw.AlphaCutoff,bool(draw.DiffuseTexture),bool(draw.OpacityTexture),value) ||
                        !geometry(draw) || !Upload(context,value,error)) { Unbind(context); return false; }
                    auto* diffuse=texture(draw.DiffuseTexture); auto* opacity=texture(draw.OpacityTexture);
                    if ((draw.DiffuseTexture && !diffuse) || (draw.OpacityTexture && !opacity)) { Unbind(context); error="Invalid DX11 OIT material texture"; return false; }
                    ID3D11ShaderResourceView* inputs[]{opaqueDepth,Depth[step.DepthRead].View.Get(),diffuse,opacity}; context->PSSetShaderResources(0,4,inputs);
                    context->RSSetState(Rasters[(draw.BackFaceCulling ? 1u : 0u)|(RealtimeWorldIsMirrored(draw.World) ? 2u : 0u)].Get());
                    context->DrawIndexed(draw.IndexCount,draw.FirstIndex,draw.BaseVertex);
                }
                Unbind(context); return Sequence.Advance(DepthPeelingEvent::EndCapture,error);
            }
            const bool compose=pass==DepthPeelingPass::Compose;
            if ((!compose && Sequence.Stage()!=DepthPeelingStage::NeedAccumulation) ||
                (compose && Sequence.Stage()!=DepthPeelingStage::NeedComposition)) { error="DX11 OIT resolve is out of order"; return false; }
            Unbind(context); BindProgram(context,compose ? 2u : 1u); context->IASetInputLayout(nullptr);
            context->OMSetDepthStencilState(NoDepth.Get(),0); context->RSSetState(Rasters[0].Get());
            auto* target=compose ? Layer.Target.Get() : Accumulation[step.AccumulationWrite].Target.Get(); context->OMSetRenderTargets(1,&target,nullptr);
            // Composition reuses layer storage only after its last accumulation.
            ID3D11ShaderResourceView* inputs[]{compose ? nullptr : Layer.View.Get(),Accumulation[step.AccumulationRead].View.Get(),sceneView,Depth[step.DepthWrite].View.Get()};
            if (!Upload(context,base,error)) { Unbind(context); return false; }
            context->PSSetShaderResources(4,4,inputs); context->Draw(3,0); Unbind(context);
            if (compose) context->CopyResource(scene,Layer.Texture.Get());
            return Sequence.Advance(compose ? DepthPeelingEvent::Compose : DepthPeelingEvent::Accumulate,error);
        }
    };
}
#endif
