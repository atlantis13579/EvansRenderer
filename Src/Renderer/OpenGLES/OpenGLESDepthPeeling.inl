// OpenGLESRHI::Impl members. GLES 3.0 uses depth-only FBOs for capture,
// depth-EQUAL material FBOs for lighting and unblended RGBA32F accumulation.
DepthPeelingFrame FramePeeling;
DepthPeelingSequence PeelingSequence;
std::array<GLuint,2> PeelingDepth{},PeelingAccumulation{},PeelingCaptureFB{},PeelingLayerFB{},PeelingAccumulationFB{};
std::array<GLuint,3> PeelingPrograms{};
GLuint PeelingLayer=0,PeelingCompositionFB=0,PeelingVAO=0,PeelingRemappedIndices=0;
int PeelingWidth=0,PeelingHeight=0;

void DestroyPeelingTargets(bool complete=false)
{
    for (auto* array:{&PeelingCaptureFB,&PeelingLayerFB,&PeelingAccumulationFB})
        for (auto& framebuffer:*array) { if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1,&framebuffer); framebuffer=0; }
    if (PeelingCompositionFB && GlDeleteFramebuffers) GlDeleteFramebuffers(1,&PeelingCompositionFB);
    PeelingCompositionFB=0;
    for (auto* array:{&PeelingDepth,&PeelingAccumulation})
        for (auto& texture:*array) { if (texture && GlDeleteTextures) GlDeleteTextures(1,&texture); texture=0; }
    if (PeelingLayer && GlDeleteTextures) GlDeleteTextures(1,&PeelingLayer);
    PeelingLayer=0; PeelingWidth=PeelingHeight=0;
    if (!complete) return;
    for (auto& program:PeelingPrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; }
    if (PeelingVAO && GlDeleteVertexArrays) GlDeleteVertexArrays(1,&PeelingVAO);
    if (PeelingRemappedIndices && GlDeleteBuffers) GlDeleteBuffers(1,&PeelingRemappedIndices);
    PeelingVAO=PeelingRemappedIndices=0; PeelingSequence.Abort(); FramePeeling={};
}
bool EnsurePeelingResources()
{
    for (unsigned phase=0;phase<3;++phase)
        if (!PeelingPrograms[phase]) PeelingPrograms[phase]=CompileStageProgram("OIT.glsl",
            phase==0 ? "#define EVANS_OIT_CAPTURE 1\n" : phase==1 ? "#define EVANS_OIT_ACCUMULATE 1\n" : "");
    if (!PeelingPrograms[0] || !PeelingPrograms[1] || !PeelingPrograms[2]) return false;
    if (!PeelingVAO) GlGenVertexArrays(1,&PeelingVAO);
    if (PeelingWidth!=CreateInfo.Width || PeelingHeight!=CreateInfo.Height) DestroyPeelingTargets();
    if (PeelingCompositionFB) return true;
    auto image=[&](GLuint& texture,GLint internal,GLenum format,GLenum type) {
        GlGenTextures(1,&texture); GlBindTexture(GL_TEXTURE_2D,texture); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,internal,CreateInfo.Width,CreateInfo.Height,0,format,type,nullptr);
    };
    image(PeelingLayer,GL_RGBA16F,GL_RGBA,GL_HALF_FLOAT);
    const GLenum color=GL_COLOR_ATTACHMENT0,none=GL_NONE_VALUE;
    auto framebuffer=[&](GLuint& output,GLuint colorImage,GLuint depthImage) {
        GlGenFramebuffers(1,&output); GlBindFramebuffer(GL_FRAMEBUFFER,output);
        if (colorImage) GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,colorImage,0);
        if (depthImage) GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,depthImage,0);
        GlDrawBuffers(1,colorImage ? &color : &none); GlReadBuffer(colorImage ? color : none);
        return GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    };
    bool valid=true;
    for (unsigned index=0;index<2;++index)
    {
        image(PeelingDepth[index],GL_DEPTH_COMPONENT32F,GL_DEPTH_COMPONENT,GL_FLOAT);
        image(PeelingAccumulation[index],GL_RGBA32F,GL_RGBA,GL_FLOAT);
        valid=framebuffer(PeelingCaptureFB[index],0,PeelingDepth[index]) && valid;
        valid=framebuffer(PeelingLayerFB[index],PeelingLayer,PeelingDepth[index]) && valid;
        valid=framebuffer(PeelingAccumulationFB[index],PeelingAccumulation[index],0) && valid;
    }
    valid=framebuffer(PeelingCompositionFB,PeelingLayer,0) && valid;
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindTexture(GL_TEXTURE_2D,0);
    if (!CheckGL("Create OIT targets") || !valid) { DestroyPeelingTargets(); return Fail("OpenGL ES OIT requires complete D32, RGBA16F and RGBA32F targets"); }
    PeelingWidth=CreateInfo.Width; PeelingHeight=CreateInfo.Height; return true;
}
bool BeginPeelingLayer(GLuint& framebuffer)
{
    if (!PeelingSequence.Advance(DepthPeelingEvent::BeginShading,LastError)) return false;
    UnbindSampledTextures(); framebuffer=PeelingLayerFB[PeelingSequence.Step().DepthWrite];
    GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    GlClearColor(0,0,0,0); GlClear(GL_COLOR_BUFFER_BIT); return CheckGL("Clear OIT layer");
}
bool ProcessPeeling(DepthPeelingPass pass)
{
    const auto step=PeelingSequence.Step(); DepthPeelingConstants base;
    if (!BuildDepthPeelingConstants(FramePeeling.ViewProjection,CreateInfo.Width,CreateInfo.Height,step.FirstLayer,FramePeeling.Settings.ShowOverflow,base))
        return Fail("Invalid OpenGL ES OIT frame constants");
    auto restore=[&] { UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindVertexArray(VertexArray);
        GlDepthMask(GL_TRUE_VALUE); GlFrontFace(GL_CCW); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); };
    auto upload=[&](GLuint program,const DepthPeelingConstants& value) {
        const auto world=ToColumnMajor(value.World),projection=ToColumnMajor(value.ViewProjection);
        GlUniformMatrix4fv(GlGetUniformLocation(program,"World"),1,GL_FALSE_VALUE,world.data());
        GlUniformMatrix4fv(GlGetUniformLocation(program,"ViewProjection"),1,GL_FALSE_VALUE,projection.data());
        UploadFloat4(GlGetUniformLocation(program,"Material"),value.Material); UploadFloat4(GlGetUniformLocation(program,"Viewport"),value.Viewport);
        GlUniform4ui(GlGetUniformLocation(program,"Options"),value.Options.X,value.Options.Y,value.Options.Z,value.Options.W);
    };
    auto bind=[&](GLuint program,const std::array<GLuint,8>& images) {
        const char* names[]{"OpaqueDepth","PreviousDepth","DiffuseMap","OpacityMap","LayerColor","PreviousAccumulation","SceneColor","CapturedDepth"};
        for (GLuint slot=0;slot<images.size();++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,images[slot]); GlUniform1i(GlGetUniformLocation(program,names[slot]),slot); }
    };
    auto texture=[&](TextureHandle handle)->GLuint {
        const auto found=Textures.find(handle.Value);
        return handle && found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? found->second.Texture : 0;
    };
    if (pass==DepthPeelingPass::Capture)
    {
        if (!PeelingSequence.Advance(DepthPeelingEvent::BeginCapture,LastError)) return false;
        UnbindSampledTextures(); GlDisable(GL_BLEND); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
        GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
        GlDepthMask(GL_TRUE_VALUE); GlClearDepthf(PeelingClearDepth);
        if (step.FirstLayer)
        {
            GlClearColor(0,0,0,1);
            for (auto framebuffer:PeelingAccumulationFB) { GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer); GlClear(GL_COLOR_BUFFER_BIT); }
            for (auto framebuffer:PeelingCaptureFB) { GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer); GlClear(GL_DEPTH_BUFFER_BIT); }
        }
        GlBindFramebuffer(GL_FRAMEBUFFER,PeelingCaptureFB[step.DepthWrite]); GlClear(GL_DEPTH_BUFFER_BIT);
        GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_LESS); GlUseProgram(PeelingPrograms[0]); GlBindVertexArray(PeelingVAO);
        for (const auto& draw:FramePeeling.Draws)
        {
            if (!draw.IndexCount) continue;
            const auto vertex=Buffers.find(draw.VertexBuffer.Value),index=Buffers.find(draw.IndexBuffer.Value);
            if (vertex==Buffers.end() || index==Buffers.end()) { restore(); return Fail("Invalid OpenGL ES OIT buffers"); }
            if (!ValidateDepthPeelingBuffers(draw,vertex->second.Description,index->second.Description,vertex->second.Bytes.size(),index->second.Bytes,LastError,vertex->second.NaniteGeometry && index->second.NaniteGeometry)) { restore(); return false; }
            GlBindBuffer(GL_ARRAY_BUFFER,vertex->second.Buffer); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,index->second.Buffer);
            const GLint components[]{3,3,4,2};
            const std::size_t offsets[]{offsetof(Vertex,Position),offsetof(Vertex,Normal),offsetof(Vertex,Color),offsetof(Vertex,TexCoord)};
            for (GLuint slot=0;slot<std::size(components);++slot)
            { GlEnableVertexAttribArray(slot); GlVertexAttribPointer(slot,components[slot],GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsets[slot])); }
            std::size_t first=draw.FirstIndex;
            if (draw.BaseVertex)
            {
                std::vector<std::uint32_t> remapped(draw.IndexCount);
                for (std::size_t item=0;item<remapped.size();++item)
                {
                    std::uint32_t original; std::memcpy(&original,index->second.Bytes.data()+(first+item)*sizeof(original),sizeof(original));
                    remapped[item]=static_cast<std::uint32_t>(std::int64_t(original)+draw.BaseVertex);
                }
                if (!PeelingRemappedIndices) GlGenBuffers(1,&PeelingRemappedIndices);
                GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,PeelingRemappedIndices);
                GlBufferData(GL_ELEMENT_ARRAY_BUFFER,remapped.size()*sizeof(std::uint32_t),remapped.data(),GL_DYNAMIC_DRAW); first=0;
            }
            DepthPeelingConstants value;
            if (!BuildDepthPeelingDrawConstants(base,draw.World,draw.Opacity,draw.AlphaCutoff,bool(draw.DiffuseTexture),bool(draw.OpacityTexture),value))
            { restore(); return Fail("Invalid OpenGL ES OIT draw constants"); }
            const auto diffuse=texture(draw.DiffuseTexture),opacity=texture(draw.OpacityTexture);
            if ((draw.DiffuseTexture && !diffuse) || (draw.OpacityTexture && !opacity)) { restore(); return Fail("Invalid OpenGL ES OIT material texture"); }
            upload(PeelingPrograms[0],value); bind(PeelingPrograms[0],{DepthTexture,PeelingDepth[step.DepthRead],diffuse,opacity,0,0,0,0});
            if (draw.BackFaceCulling) { GlEnable(GL_CULL_FACE); GlCullFace(GL_BACK); } else GlDisable(GL_CULL_FACE);
            GlFrontFace(RealtimeWorldIsMirrored(draw.World) ? GL_CW : GL_CCW);
            GlDrawElements(GL_TRIANGLES,draw.IndexCount,GL_UNSIGNED_INT,reinterpret_cast<const void*>(first*sizeof(std::uint32_t)));
        }
        const bool valid=CheckGL("Capture OIT layer"); restore();
        return valid && PeelingSequence.Advance(DepthPeelingEvent::EndCapture,LastError);
    }
    const bool compose=pass==DepthPeelingPass::Compose;
    if ((!compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedAccumulation) ||
        (compose && PeelingSequence.Stage()!=DepthPeelingStage::NeedComposition)) return Fail("OpenGL ES OIT resolve is out of order");
    UnbindSampledTextures(); GlDisable(GL_BLEND); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    GlBindFramebuffer(GL_FRAMEBUFFER,compose ? PeelingCompositionFB : PeelingAccumulationFB[step.AccumulationWrite]);
    const auto program=PeelingPrograms[compose ? 2 : 1]; GlUseProgram(program); GlBindVertexArray(VertexArray); upload(program,base);
    bind(program,{0,0,0,0,compose ? 0 : PeelingLayer,PeelingAccumulation[step.AccumulationRead],HdrTextures[0],PeelingDepth[step.DepthWrite]});
    GlDrawArrays(GL_TRIANGLES,0,3); UnbindSampledTextures();
    if (compose)
    {
        GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
        GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
    }
    const bool valid=CheckGL("Resolve OIT layer"); restore();
    return valid && PeelingSequence.Advance(compose ? DepthPeelingEvent::Compose : DepthPeelingEvent::Accumulate,LastError);
}
