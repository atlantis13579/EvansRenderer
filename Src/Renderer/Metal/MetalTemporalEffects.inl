TemporalExposureHistory ExposureHistory;
std::vector<ObjCObject> ExposureLevels;
std::array<ObjCObject,2> ExposureValues{};
std::vector<ExposureExtent> ExposureExtents;
ObjCObject ExposureMeterPipeline=nullptr,ExposureCompositePipeline=nullptr;
int ExposureWidth=0,ExposureHeight=0;
void DestroyExposureTargets()
{
    for (auto image:ExposureLevels) ReleaseObject(image);
    for (auto& image:ExposureValues) { ReleaseObject(image); image=nullptr; }
    ExposureLevels.clear(); ExposureExtents.clear(); ExposureWidth=ExposureHeight=0; ExposureHistory.Invalidate();
}
void DestroyExposure()
{
    DestroyExposureTargets(); ReleaseObject(ExposureMeterPipeline); ReleaseObject(ExposureCompositePipeline);
    ExposureMeterPipeline=ExposureCompositePipeline=nullptr;
}
bool EnsureExposure()
{
    if (!ExposureMeterPipeline) ExposureMeterPipeline=CreateTechniquePipeline("TemporalEffects.metal","VSMain","PSMain",1,{PixelFormatRGBA32Float},false);
    if (!ExposureCompositePipeline) ExposureCompositePipeline=CreateTechniquePipeline("TemporalEffects.metal","VSMain","PSMain",1,{PixelFormatRGBA16Float},false);
    if (!ExposureMeterPipeline || !ExposureCompositePipeline) return false;
    if (ExposureWidth==CreateInfo.Width && ExposureHeight==CreateInfo.Height) return true;
    DestroyExposureTargets(); ExposureExtents=BuildExposureReductionPlan(CreateInfo.Width,CreateInfo.Height);
    for (const auto extent:ExposureExtents)
        ExposureLevels.push_back(CreateNativeTexture(extent.Width,extent.Height,PixelFormatRGBA32Float,
            TextureUsageShaderRead|TextureUsageRenderTarget,"Evans exposure sum/count"));
    for (auto& value:ExposureValues) value=CreateNativeTexture(1,1,PixelFormatRGBA32Float,
        TextureUsageShaderRead|TextureUsageRenderTarget,"Evans temporal exposure EV");
    if (std::any_of(ExposureLevels.begin(),ExposureLevels.end(),[](auto image){return !image;}) || !ExposureValues[0] || !ExposureValues[1])
    { DestroyExposureTargets(); return Fail("Metal exposure FP32 target allocation failed"); }
    ExposureWidth=CreateInfo.Width; ExposureHeight=CreateInfo.Height; return true;
}
bool ResolveExposure(const TemporalEffectsDesc& desc)
{
    TemporalEffectsConstants constants;
    if (!ExposureHistory.Prepare(desc,FramePostProcessSettings.TemporalEffects,CreateInfo.Width,CreateInfo.Height,constants))
        return Fail("Exposure requires finite scene time and positive dimensions");
    const auto draw=[&](ObjCObject source,ObjCObject previous,ObjCObject destination,ExposureExtent sourceSize,ExposureExtent targetSize,Techniques::TemporalEffectsPassKind phase)
    {
        ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",0);
        Send<void,ObjCObject>(color,"setTexture:",destination); Send<void,NSUInteger>(color,"setLoadAction:",LoadActionClear); Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
        ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
        if (!encoder) return Fail("Metal exposure encoder creation failed");
        const MTLViewport viewport{0,0,double(targetSize.Width),double(targetSize.Height),0,1}; Send<void,MTLViewport>(encoder,"setViewport:",viewport);
        Send<void,ObjCObject>(encoder,"setRenderPipelineState:",phase==Techniques::TemporalEffectsPassKind::Composite?ExposureCompositePipeline:ExposureMeterPipeline);
        Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone); Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
        constants.PhaseExtentHistory.X=static_cast<unsigned>(phase); constants.PhaseExtentHistory.Y=sourceSize.Width; constants.PhaseExtentHistory.Z=sourceSize.Height;
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),0);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",source,0); Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",previous,1);
        Send<void,NSUInteger,NSUInteger,NSUInteger>(encoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
        Send<void>(encoder,"endEncoding"); ++PendingFrame.Statistics.TemporalEffectsPassCount; return true;
    };
    ObjCObject source=HdrTextures[0]; ExposureExtent size{CreateInfo.Width,CreateInfo.Height};
    for (std::size_t index=0;index<ExposureLevels.size();++index)
    {
        if (!draw(source,source,ExposureLevels[index],size,ExposureExtents[index],index?Techniques::TemporalEffectsPassKind::Reduce:Techniques::TemporalEffectsPassKind::Meter)) return false;
        source=ExposureLevels[index]; size=ExposureExtents[index];
    }
    const auto current=ExposureValues[1-ExposureHistory.ReadIndex()];
    if (!draw(source,constants.PhaseExtentHistory.W?ExposureValues[ExposureHistory.ReadIndex()]:source,current,size,{1,1},Techniques::TemporalEffectsPassKind::Adapt) ||
        !draw(HdrTextures[0],current,HdrTextures[1],{CreateInfo.Width,CreateInfo.Height},{CreateInfo.Width,CreateInfo.Height},Techniques::TemporalEffectsPassKind::Composite) ||
        !SubsurfaceCopy(HdrTextures[1],HdrTextures[0])) return false;
    ExposureHistory.Resolved=true; return true;
}
