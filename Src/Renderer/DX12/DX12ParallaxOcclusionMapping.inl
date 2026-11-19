// POM has a compact root signature: mesh CBV, twelve height constants and three
// material tables. It does not enlarge the full forward signature's 64 DWORDs.
static constexpr UINT ParallaxOcclusionMappingSamplerRoot = 5;
ComPtr<ID3D12RootSignature> ParallaxRoot;
ComPtr<ID3DBlob> ParallaxVertex,ParallaxPixel;
void DestroyParallax() { ParallaxRoot.Reset(); ParallaxVertex.Reset(); ParallaxPixel.Reset(); }
bool EnsureParallax()
{
    if (ParallaxRoot && ParallaxVertex && ParallaxPixel) return true;
    if (!ParallaxRoot)
    {
        constexpr UINT InputCount=3;
        std::array<D3D12_ROOT_PARAMETER, ParallaxOcclusionMappingSamplerRoot + 1> parameters{};
        std::array<D3D12_DESCRIPTOR_RANGE,InputCount> ranges{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor.ShaderRegister=0;
        parameters[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants={1,0,sizeof(ParallaxConstants)/sizeof(std::uint32_t)}; parameters[1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        for (UINT slot=0;slot<InputCount;++slot)
        {
            ranges[slot].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[slot].NumDescriptors=1; ranges[slot].BaseShaderRegister=slot+1;
            ranges[slot].OffsetInDescriptorsFromTableStart=D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            parameters[slot+2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[slot+2].DescriptorTable={1,&ranges[slot]}; parameters[slot+2].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
        }
        D3D12_DESCRIPTOR_RANGE materialSamplerRange{};
        DescribeMaterialSampler(parameters[ParallaxOcclusionMappingSamplerRoot], materialSamplerRange);
        D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters=static_cast<UINT>(parameters.size()); signature.pParameters=parameters.data();
        signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> bytes,errors;
        auto result=SerializeRootSignatureFunction(&signature,D3D_ROOT_SIGNATURE_VERSION_1,bytes.GetAddressOf(),errors.GetAddressOf());
        if (FAILED(result)) return FailHRESULT("Serialize POM root signature",result);
        result=Evans::Renderer::CreateCachedRootSignature(Device,0,bytes->GetBufferPointer(),bytes->GetBufferSize(),IID_PPV_ARGS(ParallaxRoot.GetAddressOf()));
        if (FAILED(result)) return FailHRESULT("Create POM root signature",result);
    }
    return LoadTechniqueShaderSource("DX12","ParallaxOcclusionMapping.hlsl",ShaderCode,LastError) &&
        CompileShader("VSMain","vs_5_0",ParallaxVertex,nullptr,true) && CompileShader("PSMain","ps_5_0",ParallaxPixel,nullptr,true);
}
bool BindParallaxDraw(const std::array<UINT,3>& descriptors)
{
    ParallaxConstants height;
    if (!EnsureParallax() || !BuildParallaxConstants(BoundPipeline.Parallax,BoundPipeline.ParallaxBoundsMinimum,BoundPipeline.ParallaxBoundsMaximum,height))
        return Fail("POM shader state or local object bounds are invalid");
    const auto draw=BuildRealtimeDrawConstants(BoundPipeline,BoundWorld,BoundViewProjection,BoundLightViewProjection,CurrentPass.Type,
        false,bool(BoundPipeline.DiffuseTexture),bool(BoundPipeline.NormalTexture),bool(BoundPipeline.OpacityTexture),true);
    std::memcpy(ConstantData+ConstantOffset,&draw,sizeof(draw));
    CommandList->SetGraphicsRootSignature(ParallaxRoot.Get());
    BindMaterialSampler(ParallaxOcclusionMappingSamplerRoot);
    CommandList->SetGraphicsRootConstantBufferView(0,ConstantBuffer->GetGPUVirtualAddress()+ConstantOffset); ConstantOffset+=sizeof(draw);
    CommandList->SetGraphicsRoot32BitConstants(1,sizeof(height)/sizeof(std::uint32_t),&height,0);
    for (UINT slot=0;slot<descriptors.size();++slot) CommandList->SetGraphicsRootDescriptorTable(slot+2,SRVArena.Gpu(descriptors[slot]));
    return true;
}
