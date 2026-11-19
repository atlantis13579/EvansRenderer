// A completed CPU publication owns a private shared-storage RGBA32F texture.
// Metal encoders snapshot each stage's constants and retain bound resources.
VoxelGIFrame FrameVoxelGI;
std::shared_ptr<const VoxelRadianceVolume> UploadedVoxelVolume;
ObjCObject VoxelVolume=nullptr;
std::array<ObjCObject,2> VoxelGIImages{},VoxelGIPipelines{};
int VoxelGIWidth=0,VoxelGIHeight=0;
bool VoxelSamplingSupported=false;
bool ProbeVoxelSamplingSupport()
{
    ObjCObject texture=CreateNativeTexture(1,1,PixelFormatRGBA32Float,TextureUsageShaderRead,"Evans voxel sampling probe");
    const Float4 empty{0,0,0,0}; const bool supported=texture && UploadTexture(texture,1,1,sizeof(empty),&empty);
    ReleaseObject(texture); return supported;
}
void DestroyVoxelGI()
{
    ReleaseObject(VoxelVolume); VoxelVolume=nullptr;
    for (auto& image:VoxelGIImages) { ReleaseObject(image); image=nullptr; }
    for (auto& pipeline:VoxelGIPipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    VoxelGIWidth=VoxelGIHeight=0; UploadedVoxelVolume.reset(); FrameVoxelGI={};
}
bool EnsureVoxelGIResources()
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants({},frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("Metal voxel GI publication is invalid");
    if (frame.Volume!=UploadedVoxelVolume)
    {
        const auto& volume=*frame.Volume;
        ObjCObject image=CreateNativeTexture(volume.Width,volume.Height,PixelFormatRGBA32Float,TextureUsageShaderRead,"Evans voxel radiance");
        if (!image) return Fail("Metal voxel radiance texture allocation failed");
        if (!UploadTexture(image,volume.Width,volume.Height,sizeof(Float4),volume.Texels.data())) { ReleaseObject(image); return false; }
        ReleaseObject(VoxelVolume); VoxelVolume=image; UploadedVoxelVolume=frame.Volume;
    }
    if (VoxelGIWidth!=CreateInfo.Width || VoxelGIHeight!=CreateInfo.Height)
    {
        std::array<ObjCObject,2> images{};
        for (auto& image:images) image=CreateNativeTexture(CreateInfo.Width,CreateInfo.Height,PixelFormatRGBA16Float,
            TextureUsageShaderRead|TextureUsageRenderTarget,"Evans voxel GI reconstruction");
        if (!images[0] || !images[1]) { for (auto image:images) ReleaseObject(image); return Fail("Metal voxel GI target allocation failed"); }
        for (auto image:VoxelGIImages) ReleaseObject(image);
        VoxelGIImages=images; VoxelGIWidth=CreateInfo.Width; VoxelGIHeight=CreateInfo.Height;
    }
    if (VoxelGIPipelines[0] && VoxelGIPipelines[1]) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","VoxelGlobalIllumination.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Metal voxel GI compilation failed: "+ErrorDescription(error));
    ObjCObject vertex=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("VSMain"));
    ObjCObject fragment=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("PSMain"));
    std::array<ObjCObject,2> pipelines{};
    if (vertex && fragment) for (std::size_t index=0;index<pipelines.size();++index)
    {
        const bool composite=index==1; ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
        Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,NSUInteger>(color,"setPixelFormat:",PixelFormatRGBA16Float); Send<void,NSUInteger>(color,"setWriteMask:",composite ? ColorWriteMaskRGB:ColorWriteMaskAll);
        Send<void,ObjCBool>(color,"setBlendingEnabled:",composite ? ObjCYes:ObjCNo);
        if (composite)
        {
            Send<void,NSUInteger>(color,"setSourceRGBBlendFactor:",BlendFactorOne); Send<void,NSUInteger>(color,"setDestinationRGBBlendFactor:",BlendFactorOne);
            Send<void,NSUInteger>(color,"setRgbBlendOperation:",BlendOperationAdd);
        }
        error=nullptr; pipelines[index]=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
        ReleaseObject(descriptor); if (!pipelines[index]) break;
    }
    ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
    if (!pipelines[0] || !pipelines[1]) { for (auto pipeline:pipelines) ReleaseObject(pipeline); return Fail("Metal voxel GI pipeline failed: "+ErrorDescription(error)); }
    for (auto pipeline:VoxelGIPipelines) ReleaseObject(pipeline); VoxelGIPipelines=pipelines; return true;
}
bool ApplyVoxelGI(const GlobalIlluminationDesc& desc)
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants(desc.ViewProjection,frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("Metal voxel GI camera is invalid");
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1};
    for (const auto pass:VoxelGIPasses)
    {
        const auto index=static_cast<std::size_t>(pass); const bool composite=pass==Techniques::VoxelGIPass::Composite;
        const ObjCObject previous=index ? VoxelGIImages[(index-1)%2]:DummyColorTexture;
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,ObjCObject>(color,"setTexture:",composite ? HdrTextures[0]:VoxelGIImages[index%2]);
        Send<void,NSUInteger>(color,"setLoadAction:",composite ? LoadActionLoad:LoadActionClear); Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
        ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!encoder) return Fail("Metal voxel GI encoder allocation failed");
        Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,ObjCObject>(encoder,"setRenderPipelineState:",VoxelGIPipelines[composite ? 1:0]);
        Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone); Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        const std::array<ObjCObject,7> textures=composite ?
            std::array<ObjCObject,7>{DummyShadowTexture,DummyColorTexture,DummyColorTexture,DummyColorTexture,DummyColorTexture,previous,DummyColorTexture} :
            std::array<ObjCObject,7>{DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3],VoxelVolume,previous,HdrTextures[0]};
        for (std::size_t slot=0;slot<textures.size();++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",textures[slot],slot);
        constants.Options.X=static_cast<std::uint32_t>(pass); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3); Send<void>(encoder,"endEncoding");
        PendingFrame.VoxelGIPasses.push_back(pass); ++PendingFrame.Statistics.GlobalIlluminationPassCount;
    }
    GI.Resolved=true; return true;
}
