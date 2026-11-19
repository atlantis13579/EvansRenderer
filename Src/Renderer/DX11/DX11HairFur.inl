// Fiber storage is immutable between synchronized frames. A separate layout
// preserves the ordinary mesh ABI; both HDR and receiver-mask attachments load.
HairFurExecution Hair;
HairFurFrame UploadedHair;
std::array<ID3D11ShaderResourceView*,HairShadowMapCount> HairShadowViews{};
bool PrepareHairShadowReceiver(const HairFurDesc& desc,HairFurConstants& constants)
{
    SetHairFurAnimation(Hair.Frame,Hair.Frame.SampleTimeSeconds,constants);
    constants.MotionOptions.Y=Hair.LumenReceiver?1u:0u;
    if (!BuildHairContactControl(desc.ViewProjection,*Hair.Frame.Lights,FrameContact.Enabled && !Hair.Frame.Geometry->Indices.empty(),
        Hair.Contact,ContactGenerated && ContactReceiverActive && ContactShadows.View(),FrameSampleCount,constants.ContactControl,LastError)) return false;
    HairShadowViews={}; std::array<HairShadowMapReadiness,HairShadowMapCount> maps{};
    const bool requested=desc.SamplesShadowMap && !Hair.Frame.Geometry->Indices.empty();
    if (requested && Hair.Frame.Lights->ShadowBegin!=Hair.Frame.Lights->ShadowEnd)
    {
        const auto count=FrameCascades.Frame.CascadeCount,first=count?1u:0u;
        for (unsigned slot=first;slot<first+(count?count:1u) && slot<HairShadowMapCount;++slot)
        {
            const auto handle=slot?FrameCascades.Maps[slot-1]:LastShadowTexture;
            const auto& projection=slot?FrameCascades.Frame.Cascades[slot-1].LightViewProjection:desc.LightViewProjection;
            const auto found=Textures.find(handle.Value);
            if (found==Textures.end() || found->second.Description.Format!=TextureFormat::Depth32) continue;
            const auto& texture=found->second;
            maps[slot]={std::uint32_t(texture.Description.Width),std::uint32_t(texture.Description.Height),Hair.HasShadowView(handle,projection),false};
            if (!maps[slot].DepthReady) continue;
            HairShadowViews[slot]=FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM?VSM.View(handle):texture.ShaderResourceView.Get();
            maps[slot].MomentsReady=FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM && HairShadowViews[slot];
        }
    }
    return BuildHairShadowReceiverConstants({desc.LightViewProjection,requested},*Hair.Frame.Lights,FrameCascades,FrameShadowFilter,maps,constants.Shadows,LastError);
}
ComPtr<ID3D11Buffer> HairVertices,HairIndices,HairLights,HairUniforms;
ComPtr<ID3D11ShaderResourceView> HairLightView;
ComPtr<ID3D11VertexShader> HairVertexProgram;
std::array<ComPtr<ID3D11PixelShader>,6> HairPixels;
ComPtr<ID3D11InputLayout> HairLayout;
ComPtr<ID3D11RasterizerState> HairRaster,HairShadowRaster;
ComPtr<ID3D11DepthStencilState> HairDepth,HairVelocityDepth;
void DestroyHairFur()
{
    Hair={}; UploadedHair={}; HairVertices.Reset(); HairIndices.Reset(); HairLights.Reset(); HairUniforms.Reset();
    HairLightView.Reset(); HairVertexProgram.Reset(); HairPixels={}; HairLayout.Reset(); HairRaster.Reset(); HairShadowRaster.Reset(); HairDepth.Reset(); HairVelocityDepth.Reset();
}
bool EnsureHairFur()
{
    const auto& frame=Hair.Frame;
    if (UploadedHair.Geometry!=frame.Geometry || UploadedHair.Lights!=frame.Lights)
    {
        if (!ValidateHairFurPublication(frame,LastError)) return false;
        ComPtr<ID3D11Buffer> vertices=HairVertices,indices=HairIndices,lights=HairLights;
        ComPtr<ID3D11ShaderResourceView> view=HairLightView;
        const auto create=[&](const void* data,std::size_t bytes,UINT flags,UINT stride,ComPtr<ID3D11Buffer>& buffer)
        {
            if (bytes>UINT_MAX) return Fail("DX11 hair buffer exceeds native byte limits");
            D3D11_BUFFER_DESC desc{}; desc.ByteWidth=static_cast<UINT>(bytes); desc.Usage=D3D11_USAGE_IMMUTABLE; desc.BindFlags=flags;
            if (stride) { desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; desc.StructureByteStride=stride; }
            const D3D11_SUBRESOURCE_DATA initial{data,0,0};
            return SUCCEEDED(Device->CreateBuffer(&desc,&initial,buffer.ReleaseAndGetAddressOf())) || Fail("DX11 hair buffer allocation failed");
        };
        if (UploadedHair.Geometry!=frame.Geometry)
        {
            vertices.Reset(); indices.Reset(); const auto& geometry=*frame.Geometry;
            if (!geometry.Indices.empty() && (!create(geometry.Vertices.data(),geometry.Vertices.size()*sizeof(HairVertex),D3D11_BIND_VERTEX_BUFFER,0,vertices) ||
                !create(geometry.Indices.data(),geometry.Indices.size()*sizeof(std::uint32_t),D3D11_BIND_INDEX_BUFFER,0,indices))) return false;
        }
        if (UploadedHair.Lights!=frame.Lights)
        {
            const HairLightSample empty{}; const auto& samples=frame.Lights->Samples;
            if (!create(samples.empty()?&empty:samples.data(),std::max(samples.size(),std::size_t{1})*sizeof(HairLightSample),D3D11_BIND_SHADER_RESOURCE,sizeof(HairLightSample),lights)) return false;
            D3D11_SHADER_RESOURCE_VIEW_DESC desc{}; desc.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
            desc.Buffer.NumElements=static_cast<UINT>(std::max(samples.size(),std::size_t{1}));
            if (FAILED(Device->CreateShaderResourceView(lights.Get(),&desc,view.ReleaseAndGetAddressOf()))) return Fail("DX11 hair light view allocation failed");
        }
        HairVertices=vertices; HairIndices=indices; HairLights=lights; HairLightView=view; UploadedHair=frame;
    }
    if (HairVertexProgram && HairLayout && HairRaster && HairShadowRaster && HairDepth && HairVelocityDepth && HairUniforms) return true;
    if (!LoadTechniqueShaderSource("DX11","HairFur.hlsl",ShaderCode,LastError)) return false;
    ComPtr<ID3DBlob> vertex;
    if (!CompileShader("VSMain","vs_5_0",vertex,nullptr,true) ||
        FAILED(Device->CreateVertexShader(vertex->GetBufferPointer(),vertex->GetBufferSize(),nullptr,HairVertexProgram.ReleaseAndGetAddressOf()))) return Fail("DX11 hair vertex creation failed");
    const D3D11_INPUT_ELEMENT_DESC elements[]{
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(HairVertex,Position),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(HairVertex,Tangent),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Reflectance),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",1,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Emission),D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(HairVertex,Ribbon),D3D11_INPUT_PER_VERTEX_DATA,0}};
    if (FAILED(Device->CreateInputLayout(elements,static_cast<UINT>(std::size(elements)),vertex->GetBufferPointer(),vertex->GetBufferSize(),HairLayout.ReleaseAndGetAddressOf()))) return Fail("DX11 hair vertex layout creation failed");
    D3D11_RASTERIZER_DESC raster{}; raster.FillMode=D3D11_FILL_SOLID; raster.CullMode=D3D11_CULL_NONE; raster.DepthClipEnable=TRUE; raster.MultisampleEnable=TRUE;
    D3D11_DEPTH_STENCIL_DESC depth{}; depth.DepthEnable=TRUE; depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL; depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    D3D11_BUFFER_DESC buffer{}; buffer.ByteWidth=sizeof(HairFurConstants); buffer.Usage=D3D11_USAGE_DYNAMIC; buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER; buffer.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    if (FAILED(Device->CreateRasterizerState(&raster,HairRaster.ReleaseAndGetAddressOf()))) return Fail("DX11 hair raster state creation failed");
    raster.DepthBias=ShadowDepthBiasUnits; raster.DepthBiasClamp=ShadowDepthBiasClamp; raster.SlopeScaledDepthBias=ShadowSlopeScaledDepthBias;
    auto readOnlyDepth=depth; readOnlyDepth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
    if (FAILED(Device->CreateDepthStencilState(&readOnlyDepth,HairVelocityDepth.ReleaseAndGetAddressOf()))) return Fail("DX11 fiber velocity depth state creation failed");
    return (SUCCEEDED(Device->CreateRasterizerState(&raster,HairShadowRaster.ReleaseAndGetAddressOf())) &&
        SUCCEEDED(Device->CreateDepthStencilState(&depth,HairDepth.ReleaseAndGetAddressOf())) &&
        SUCCEEDED(Device->CreateBuffer(&buffer,nullptr,HairUniforms.ReleaseAndGetAddressOf()))) || Fail("DX11 hair drawing state allocation failed");
}

bool DrawHairGeometry(const HairFurConstants& constants,HairFurRasterPhase phase)
{
    if (Hair.Frame.Geometry->Indices.empty()) return true;
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(Context->Map(HairUniforms.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) return Fail("DX11 hair constants mapping failed");
    std::memcpy(mapped.pData,&constants,sizeof(constants)); Context->Unmap(HairUniforms.Get(),0);
    ID3D11ShaderResourceView* empty[16]{}; Context->PSSetShaderResources(0,16,empty);
    const bool shadow=phase==HairFurRasterPhase::ShadowDepth || phase==HairFurRasterPhase::ShadowMoments;
    const auto pixel=phase==HairFurRasterPhase::Velocity?4:phase==HairFurRasterPhase::Surface?(Hair.UsesContactReceiver()?5:0):phase==HairFurRasterPhase::Capture?3:phase==HairFurRasterPhase::ShadowMoments?2:1;
    // Compile only the phase being drawn; unused capture/contact variants must
    // not delay the first visible groom.
    if (!HairPixels[pixel])
    {
        if (!LoadTechniqueShaderSource("DX11","HairFur.hlsl",ShaderCode,LastError)) return false;
        const char* definitions[]{nullptr,"EVANS_HAIR_DEPTH_ONLY","EVANS_HAIR_VSM_SHADOW","EVANS_HAIR_CAPTURE","EVANS_HAIR_VELOCITY","EVANS_HAIR_CONTACT_RECEIVER"};
        const D3D_SHADER_MACRO macros[]{{definitions[pixel],"1"},{nullptr,nullptr}}; ComPtr<ID3DBlob> code;
        if (!CompileShader("PSMain","ps_5_0",code,definitions[pixel]?macros:nullptr,true) ||
            FAILED(Device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,HairPixels[pixel].ReleaseAndGetAddressOf()))) return Fail("DX11 hair pixel variant creation failed");
    }
    Context->RSSetState(shadow?HairShadowRaster.Get():HairRaster.Get()); Context->OMSetDepthStencilState(phase==HairFurRasterPhase::Velocity?HairVelocityDepth.Get():HairDepth.Get(),0); Context->OMSetBlendState(nullptr,nullptr,UINT_MAX);
    Context->IASetInputLayout(HairLayout.Get()); Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const UINT stride=sizeof(HairVertex),offset=0; auto* vertices=HairVertices.Get(); Context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
    Context->IASetIndexBuffer(HairIndices.Get(),DXGI_FORMAT_R32_UINT,0);
    Context->VSSetShader(HairVertexProgram.Get(),nullptr,0); Context->PSSetShader(HairPixels[pixel].Get(),nullptr,0);
    auto* uniform=HairUniforms.Get(); Context->VSSetConstantBuffers(0,1,&uniform); Context->PSSetConstantBuffers(0,1,&uniform);
    auto* lights=HairLightView.Get(); Context->PSSetShaderResources(0,1,&lights);
    if (phase==HairFurRasterPhase::Surface) Context->PSSetShaderResources(1,HairShadowMapCount,HairShadowViews.data());
    if (phase==HairFurRasterPhase::Surface && constants.ContactControl.X)
    { auto* contact=ContactShadows.View(); Context->PSSetShaderResources(6,1,&contact); }
    Context->DrawIndexed(static_cast<UINT>(Hair.Frame.Geometry->Indices.size()),0,0);
    Context->PSSetShaderResources(0,16,empty); return true;
}
bool RenderHairFur(const HairFurConstants& constants)
{
    ID3D11ShaderResourceView* empty[16]{}; Context->PSSetShaderResources(0,16,empty); Context->OMSetRenderTargets(0,nullptr,nullptr);
    ID3D11RenderTargetView* targets[]{FrameSampleCount>1?Multisampling.Color.Target.Get():HdrSurfaces[HdrScene].Target.Get(),
        FrameSampleCount>1?Multisampling.Materials[1].Target.Get():GBufferSurfaces[1].Target.Get()};
    Context->OMSetRenderTargets(static_cast<UINT>(std::size(targets)),targets,FrameSampleCount>1?Multisampling.DepthTarget.Get():DepthTarget.Get());
    const D3D11_VIEWPORT viewport{0,0,float(CreateInfo.Width),float(CreateInfo.Height),0,1}; Context->RSSetViewports(1,&viewport);
    const bool result=DrawHairGeometry(constants,HairFurRasterPhase::Surface); Context->OMSetRenderTargets(0,nullptr,nullptr); return result;
}
bool ReplayHairFur(const HairFurConstants& constants,HairFurRasterPhase phase) { return DrawHairGeometry(constants,phase); }
