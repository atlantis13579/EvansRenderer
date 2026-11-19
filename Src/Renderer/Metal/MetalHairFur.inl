// Metal reads packed ribbon vertices directly from buffer zero. Constants are
// copied into the encoder, while immutable emitter storage remains at slot two.
// Depth replay requires invariant vertex positions and preserveInvariance:
// https://developer.apple.com/documentation/metal/mtlcompileoptions/preserveinvariance
HairFurExecution Hair;
HairFurFrame UploadedHair;
std::array<ObjCObject,HairShadowMapCount> HairShadowTextures{};
unsigned HairPipelineKey(HairFurRasterPhase phase,unsigned sampleCount) const
{
    constexpr unsigned MomentReceiverBit=1u<<16;
    return HairFurPipelineKey(phase,sampleCount,Hair.UsesContactReceiver())|(phase==HairFurRasterPhase::Surface && FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM?MomentReceiverBit:0);
}
bool PrepareHairShadowReceiver(const HairFurDesc& desc,HairFurConstants& constants)
{
    SetHairFurAnimation(Hair.Frame,Hair.Frame.SampleTimeSeconds,constants);
    constants.MotionOptions.Y=Hair.LumenReceiver?1u:0u;
    if (!BuildHairContactControl(desc.ViewProjection,*Hair.Frame.Lights,FrameContact.Enabled && !Hair.Frame.Geometry->Indices.empty(),
        Hair.Contact,ContactGenerated && ContactReceiverActive && ContactProducer.Result(),FrameSampleCount,constants.ContactControl,LastError)) return false;
    const bool moments=FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM;
    HairShadowTextures.fill(moments?DummyColorTexture:DummyShadowTexture);
    std::array<HairShadowMapReadiness,HairShadowMapCount> maps{};
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
            HairShadowTextures[slot]=texture.Texture;
            if (moments)
            {
                if (!MomentImage(handle,HairShadowTextures[slot])) return false;
                maps[slot].MomentsReady=true;
            }
        }
    }
    return BuildHairShadowReceiverConstants({desc.LightViewProjection,requested},*Hair.Frame.Lights,FrameCascades,FrameShadowFilter,maps,constants.Shadows,LastError);
}
ObjCObject HairVertices=nullptr,HairIndices=nullptr,HairLights=nullptr,HairDepth=nullptr,HairVelocityDepth=nullptr;
std::array<std::unordered_map<unsigned,ObjCObject>,HairFurPhaseCount> HairPipelines;
void DestroyHairFur()
{
    for (auto& pipelines:HairPipelines) for (auto& entry:pipelines) ReleaseObject(entry.second);
    for (auto object:{HairVertices,HairIndices,HairLights,HairDepth,HairVelocityDepth}) ReleaseObject(object);
    HairVertices=HairIndices=HairLights=HairDepth=HairVelocityDepth=nullptr; for (auto& pipelines:HairPipelines) pipelines.clear(); Hair={}; UploadedHair={};
}
bool EnsureHairFur()
{
    const auto& frame=Hair.Frame;
    if (UploadedHair.Geometry!=frame.Geometry || UploadedHair.Lights!=frame.Lights)
    {
        if (!ValidateHairFurPublication(frame,LastError)) return false;
        ObjCObject vertices=nullptr,indices=nullptr,lights=nullptr;
        const auto cleanup=[&](){ReleaseObject(vertices); ReleaseObject(indices); ReleaseObject(lights);};
        const auto create=[&](const void* data,std::size_t bytes,ObjCObject& buffer)
        {
            buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
            if (!buffer) return Fail("Metal hair upload allocation failed");
            void* mapped=Send<void*>(buffer,"contents");
            if (!mapped) return Fail("Metal hair upload has no shared contents");
            std::memcpy(mapped,data,bytes); return true;
        };
        if (UploadedHair.Geometry!=frame.Geometry && !frame.Geometry->Indices.empty())
        {
            if (!create(frame.Geometry->Vertices.data(),frame.Geometry->Vertices.size()*sizeof(HairVertex),vertices) ||
                !create(frame.Geometry->Indices.data(),frame.Geometry->Indices.size()*sizeof(std::uint32_t),indices)) { cleanup(); return false; }
        }
        if (UploadedHair.Lights!=frame.Lights)
        {
            const HairLightSample empty{}; const auto& samples=frame.Lights->Samples;
            if (!create(samples.empty()?&empty:samples.data(),std::max(samples.size(),std::size_t{1})*sizeof(HairLightSample),lights)) { cleanup(); return false; }
        }
        if (UploadedHair.Geometry!=frame.Geometry)
        { ReleaseObject(HairVertices); ReleaseObject(HairIndices); HairVertices=vertices; HairIndices=indices; }
        if (UploadedHair.Lights!=frame.Lights) { ReleaseObject(HairLights); HairLights=lights; }
        UploadedHair=frame;
    }
    if (!HairDepth) HairDepth=CreateDepthState(CompareFunctionLessEqual,true);
    if (!HairVelocityDepth) HairVelocityDepth=CreateDepthState(CompareFunctionLessEqual,false);
    if (!HairDepth || !HairVelocityDepth) return Fail("Metal hair depth state creation failed");
    return true;
}
bool EnsureHairPipeline(HairFurRasterPhase phase,unsigned sampleCount)
{
    auto& pipelines=HairPipelines[HairFurPhaseIndex(phase)];
    const auto key=HairPipelineKey(phase,sampleCount);
    if (pipelines.count(key)) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","HairFur.metal",source,LastError)) return false;
    if (phase!=HairFurRasterPhase::Surface)
        source=std::string(phase==HairFurRasterPhase::Velocity?"#define EVANS_HAIR_VELOCITY 1\n":phase==HairFurRasterPhase::Capture?"#define EVANS_HAIR_CAPTURE 1\n":phase==HairFurRasterPhase::ShadowMoments?"#define EVANS_HAIR_VSM_SHADOW 1\n":"#define EVANS_HAIR_DEPTH_ONLY 1\n")+source;
    else if (FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM) source="#define EVANS_HAIR_MOMENT_RECEIVER 1\n"+source;
    if (phase==HairFurRasterPhase::Surface && Hair.UsesContactReceiver())
        source=std::string(sampleCount>1?"#define EVANS_HAIR_CONTACT_SAMPLES 1\n":"")+"#define EVANS_HAIR_CONTACT_RECEIVER 1\n"+source;
    ObjCObject options=PositionInvariantCompileOptions(),error=nullptr;
    if (!SupportsPositionInvariance(options)) { ReleaseObject(options); return Fail("Metal hair replay requires invariant vertex position compilation"); }
    Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options);
    if (!library) return Fail("Metal hair compilation failed: "+ErrorDescription(error));
    ObjCObject vertex=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("VSMain"));
    ObjCObject fragment=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString("PSMain"));
    ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPipelineDescriptor"),"new");
    Send<void,ObjCObject>(descriptor,"setVertexFunction:",vertex); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
    Send<void,NSUInteger>(descriptor,"setSampleCount:",sampleCount); Send<void,NSUInteger>(descriptor,"setDepthAttachmentPixelFormat:",PixelFormatDepth32Float);
    const NSUInteger colorCount=phase==HairFurRasterPhase::Surface?2:(phase==HairFurRasterPhase::Capture || phase==HairFurRasterPhase::ShadowMoments || phase==HairFurRasterPhase::Velocity)?1:0;
    for (NSUInteger index=0;index<colorCount;++index)
    {
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",index);
        Send<void,NSUInteger>(color,"setPixelFormat:",phase==HairFurRasterPhase::ShadowMoments?PixelFormatRG32Float:PixelFormatRGBA16Float); Send<void,NSUInteger>(color,"setWriteMask:",ColorWriteMaskAll);
        Send<void,ObjCBool>(color,"setBlendingEnabled:",ObjCNo);
    }
    ObjCObject pipeline=nullptr;
    if (vertex && fragment) pipeline=Evans::Renderer::CreateCachedMetalRenderPipeline(Device,descriptor,&error);
    ReleaseObject(descriptor); ReleaseObject(vertex); ReleaseObject(fragment); ReleaseObject(library);
    if (!pipeline) return Fail("Metal hair pipeline creation failed: "+ErrorDescription(error));
    pipelines.emplace(key,pipeline); return true;
}
bool DrawHairGeometry(ObjCObject encoder,const HairFurConstants& constants,HairFurRasterPhase phase,unsigned sampleCount)
{
    if (Hair.Frame.Geometry->Indices.empty()) return true;
    // Replay phases are compiled on demand, including after a sample-count change.
    if (!EnsureHairPipeline(phase,sampleCount)) return false;
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",HairPipelines[HairFurPhaseIndex(phase)].at(HairPipelineKey(phase,sampleCount)));
    // Match the existing Metal opaque caster's normalized raster bias.
    constexpr float CasterConstantBias=0.0015f,CasterSlopeBias=2.0f,CasterBiasClamp=0.01f;
    const bool shadow=phase==HairFurRasterPhase::ShadowDepth || phase==HairFurRasterPhase::ShadowMoments;
    Send<void,float,float,float>(encoder,"setDepthBias:slopeScale:clamp:",shadow?CasterConstantBias:0,shadow?CasterSlopeBias:0,shadow?CasterBiasClamp:0);
    Send<void,ObjCObject>(encoder,"setDepthStencilState:",phase==HairFurRasterPhase::Velocity?HairVelocityDepth:HairDepth); Send<void,NSUInteger>(encoder,"setCullMode:",CullModeNone);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",HairVertices,0,0);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&constants,sizeof(constants),1);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setFragmentBuffer:offset:atIndex:",HairLights,0,2);
    if (phase==HairFurRasterPhase::Surface)
        for (NSUInteger slot=0;slot<HairShadowMapCount;++slot) Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",HairShadowTextures[slot],slot);
    if (phase==HairFurRasterPhase::Surface && constants.ContactControl.X)
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",ContactProducer.Result(),HairShadowMapCount);
    Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
        PrimitiveTypeTriangle,Hair.Frame.Geometry->Indices.size(),IndexTypeUInt32,HairIndices,0);
    return true;
}
bool RenderHairFur(const HairFurConstants& constants)
{
    if (Hair.Frame.Geometry->Indices.empty()) return true;
    if (!EnsureHairPipeline(HairFurRasterPhase::Surface,FrameSampleCount)) return false;
    ObjCObject descriptor=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor");
    const ObjCObject textures[]{FrameSampleCount>1?Multisample.Colors[SampleColor]:HdrTextures[0],
        FrameSampleCount>1?Multisample.Colors[SampleMaterialFirst+1]:GBufferTextures[1]};
    for (NSUInteger index=0;index<std::size(textures);++index)
    {
        ObjCObject color=Send<ObjCObject,NSUInteger>(Send<ObjCObject>(descriptor,"colorAttachments"),"objectAtIndexedSubscript:",index);
        Send<void,ObjCObject>(color,"setTexture:",textures[index]); Send<void,NSUInteger>(color,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(color,"setStoreAction:",StoreActionStore);
    }
    ObjCObject depth=Send<ObjCObject>(descriptor,"depthAttachment");
    Send<void,ObjCObject>(depth,"setTexture:",FrameSampleCount>1?Multisample.Depth:DepthTexture);
    Send<void,NSUInteger>(depth,"setLoadAction:",LoadActionLoad); Send<void,NSUInteger>(depth,"setStoreAction:",StoreActionStore);
    ObjCObject encoder=Send<ObjCObject,ObjCObject>(CommandBuffer,"renderCommandEncoderWithDescriptor:",descriptor);
    if (!encoder) return Fail("Metal hair render encoder allocation failed");
    const MTLViewport viewport{0,0,double(CreateInfo.Width),double(CreateInfo.Height),0,1}; Send<void,MTLViewport>(encoder,"setViewport:",viewport);
    const bool result=DrawHairGeometry(encoder,constants,HairFurRasterPhase::Surface,FrameSampleCount); Send<void>(encoder,"endEncoding"); return result;
}
bool ReplayHairFur(const HairFurConstants& constants,HairFurRasterPhase phase)
{ return DrawHairGeometry(RenderEncoder,constants,phase,(HairFurCameraDepthPass(CurrentPass.Type) || CurrentPass.Type==PassType::Velocity)?FrameSampleCount:1); }
