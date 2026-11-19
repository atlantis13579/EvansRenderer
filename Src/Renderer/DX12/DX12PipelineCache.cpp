#include "Renderer/DX12/DX12PipelineCache.h"
#include "Renderer/ShaderCache.h"
#if defined(_WIN32)
#include <dxgi1_4.h>
#include <wrl/client.h>
namespace Evans::Renderer
{
    using Microsoft::WRL::ComPtr;
    namespace
    {
        constexpr GUID RootIdentity={0xb05c03df,0xfae8,0x417c,{0xa9,0xd2,0x51,0xc8,0x20,0x8f,0xe0,0xa5}};
        bool Common(ShaderCache::Key& key,ID3D12Device* device,ID3D12RootSignature* root)
        {
            ShaderCache::Digest digest{};UINT size=UINT(digest.size());
            if(root && (FAILED(root->GetPrivateData(RootIdentity,&size,digest.data())) || size!=digest.size()))return false;
            key.Value(root!=nullptr).Add(digest.data(),digest.size());
            const auto luid=device->GetAdapterLuid();key.Value(luid.LowPart).Value(luid.HighPart);
            ComPtr<IDXGIFactory4> factory;ComPtr<IDXGIAdapter1> adapter;
            if(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && SUCCEEDED(factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter))))
            {
                DXGI_ADAPTER_DESC1 description{};LARGE_INTEGER version{};
                adapter->GetDesc1(&description);adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice),&version);
                key.Value(description.VendorId).Value(description.DeviceId).Value(description.SubSysId).Value(description.Revision).Value(version.QuadPart);
            }
            return true;
        }
        void Shader(ShaderCache::Key& key,const D3D12_SHADER_BYTECODE& shader)
        { key.Add(shader.pShaderBytecode,shader.BytecodeLength); }
        void Face(ShaderCache::Key& key,const D3D12_DEPTH_STENCILOP_DESC& face)
        { key.Value(face.StencilFailOp).Value(face.StencilDepthFailOp).Value(face.StencilPassOp).Value(face.StencilFunc); }
        bool GraphicsKey(ShaderCache::Key& key,ID3D12Device* device,const D3D12_GRAPHICS_PIPELINE_STATE_DESC& d)
        {
            if(!Common(key,device,d.pRootSignature))return false;
            Shader(key,d.VS);Shader(key,d.PS);Shader(key,d.DS);Shader(key,d.HS);Shader(key,d.GS);
            key.Value(d.StreamOutput.NumEntries).Value(d.StreamOutput.NumStrides).Value(d.StreamOutput.RasterizedStream);
            for(UINT i=0;i<d.StreamOutput.NumEntries;++i)
            {
                const auto& e=d.StreamOutput.pSODeclaration[i];
                key.Value(e.Stream).Add(e.SemanticName?e.SemanticName:"").Value(e.SemanticIndex).Value(e.StartComponent).Value(e.ComponentCount).Value(e.OutputSlot);
            }
            for(UINT i=0;i<d.StreamOutput.NumStrides;++i)key.Value(d.StreamOutput.pBufferStrides[i]);
            key.Value(d.BlendState.AlphaToCoverageEnable).Value(d.BlendState.IndependentBlendEnable);
            for(const auto& b:d.BlendState.RenderTarget)
                key.Value(b.BlendEnable).Value(b.LogicOpEnable).Value(b.SrcBlend).Value(b.DestBlend).Value(b.BlendOp)
                    .Value(b.SrcBlendAlpha).Value(b.DestBlendAlpha).Value(b.BlendOpAlpha).Value(b.LogicOp).Value(b.RenderTargetWriteMask);
            key.Value(d.SampleMask);const auto& r=d.RasterizerState;
            key.Value(r.FillMode).Value(r.CullMode).Value(r.FrontCounterClockwise).Value(r.DepthBias).Value(r.DepthBiasClamp)
                .Value(r.SlopeScaledDepthBias).Value(r.DepthClipEnable).Value(r.MultisampleEnable).Value(r.AntialiasedLineEnable)
                .Value(r.ForcedSampleCount).Value(r.ConservativeRaster);
            const auto& z=d.DepthStencilState;
            key.Value(z.DepthEnable).Value(z.DepthWriteMask).Value(z.DepthFunc).Value(z.StencilEnable).Value(z.StencilReadMask).Value(z.StencilWriteMask);
            Face(key,z.FrontFace);Face(key,z.BackFace);key.Value(d.InputLayout.NumElements);
            for(UINT i=0;i<d.InputLayout.NumElements;++i)
            {
                const auto& e=d.InputLayout.pInputElementDescs[i];key.Add(e.SemanticName?e.SemanticName:"").Value(e.SemanticIndex)
                    .Value(e.Format).Value(e.InputSlot).Value(e.AlignedByteOffset).Value(e.InputSlotClass).Value(e.InstanceDataStepRate);
            }
            key.Value(d.IBStripCutValue).Value(d.PrimitiveTopologyType).Value(d.NumRenderTargets);
            for(auto format:d.RTVFormats)key.Value(format);
            key.Value(d.DSVFormat).Value(d.SampleDesc.Count).Value(d.SampleDesc.Quality).Value(d.NodeMask).Value(d.Flags);
            return true;
        }
        template<class Description,class Create> HRESULT Pipeline(std::string_view kind,const ShaderCache::Digest& key,
            const Description& original,REFIID iid,void** output,Create create)
        {
            auto description=original;std::vector<std::uint8_t> bytes;
            const bool cached=ShaderCache::Read(kind,key,bytes);
            bool rebuilt=!cached;
            if(cached)description.CachedPSO={bytes.data(),bytes.size()};
            auto status=create(description,iid,output);
            if(FAILED(status) && cached)
            {
                // Drivers validate the entire descriptor and may reject blobs
                // after a driver/runtime update. Rebuild without cached data.
                rebuilt=true;
                description.CachedPSO={};status=create(description,iid,output);
            }
            if(rebuilt && SUCCEEDED(status) && output && *output)
            {
                ComPtr<ID3D12PipelineState> pipeline;ComPtr<ID3DBlob> blob;
                if(SUCCEEDED(static_cast<IUnknown*>(*output)->QueryInterface(IID_PPV_ARGS(&pipeline))) &&
                    SUCCEEDED(pipeline->GetCachedBlob(&blob)))
                    ShaderCache::Write(kind,key,blob->GetBufferPointer(),blob->GetBufferSize());
            }
            return status;
        }
    }
    HRESULT CreateCachedRootSignature(ID3D12Device* device,UINT nodeMask,const void* bytes,SIZE_T size,REFIID iid,void** result)
    {
        const auto status=device->CreateRootSignature(nodeMask,bytes,size,iid,result);
        if(SUCCEEDED(status) && result && *result)
        {
            ComPtr<ID3D12RootSignature> root;
            if(SUCCEEDED(static_cast<IUnknown*>(*result)->QueryInterface(IID_PPV_ARGS(&root))))
            { const auto digest=ShaderCache::Hash(bytes,size);root->SetPrivateData(RootIdentity,UINT(digest.size()),digest.data()); }
        }
        return status;
    }
    HRESULT CreateCachedGraphicsPipelineState(ID3D12Device* device,const D3D12_GRAPHICS_PIPELINE_STATE_DESC* description,REFIID iid,void** result)
    {
        ShaderCache::Key key("dx12-graphics-pso-v1");
        if(!description || !GraphicsKey(key,device,*description))return device->CreateGraphicsPipelineState(description,iid,result);
        return Pipeline("dx12-graphics-pso",key.Finish(),*description,iid,result,
            [device](const auto& d,REFIID id,void** out){return device->CreateGraphicsPipelineState(&d,id,out);});
    }
    HRESULT CreateCachedComputePipelineState(ID3D12Device* device,const D3D12_COMPUTE_PIPELINE_STATE_DESC* description,REFIID iid,void** result)
    {
        ShaderCache::Key key("dx12-compute-pso-v1");
        if(!description || !Common(key,device,description->pRootSignature))return device->CreateComputePipelineState(description,iid,result);
        Shader(key,description->CS);key.Value(description->NodeMask).Value(description->Flags);
        return Pipeline("dx12-compute-pso",key.Finish(),*description,iid,result,
            [device](const auto& d,REFIID id,void** out){return device->CreateComputePipelineState(&d,id,out);});
    }
    HRESULT CreateCachedMeshPipelineState(ID3D12Device2* device,const D3D12_PIPELINE_STATE_STREAM_DESC& stream,
        D3D12_CACHED_PIPELINE_STATE& cached,const D3D12_GRAPHICS_PIPELINE_STATE_DESC& state,
        const D3D12_SHADER_BYTECODE& mesh,const D3D12_SHADER_BYTECODE& task,REFIID iid,void** result)
    {
        ShaderCache::Key key("dx12-mesh-pso-v1");
        if(!GraphicsKey(key,device,state))return device->CreatePipelineState(&stream,iid,result);
        Shader(key,mesh);Shader(key,task);
        const auto status=Pipeline("dx12-mesh-pso",key.Finish(),state,iid,result,
            [&](const auto& description,REFIID id,void** out) {
                cached=description.CachedPSO;return device->CreatePipelineState(&stream,id,out);
            });
        cached={};return status;
    }
}
#endif
