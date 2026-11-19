GLuint LumenHairIndirect=0,LumenHairSamples=0,LumenHairFramebuffer=0,LumenHairResolvedFramebuffer=0;
GLuint LumenHairProgram=0,LumenHairComposite=0,LumenHairUniform=0;
int LumenHairWidth=0,LumenHairHeight=0;
unsigned LumenHairSampleCount=0;
void DestroyLumenHairReceiver()
{
    const GLuint images[]{LumenHairIndirect,LumenHairSamples},targets[]{LumenHairFramebuffer,LumenHairResolvedFramebuffer};
    // Shutdown also runs when loading EGL fails before any GL entry point is available.
    if (LumenHairIndirect || LumenHairSamples) GlDeleteTextures(2,images);
    if (LumenHairFramebuffer || LumenHairResolvedFramebuffer) GlDeleteFramebuffers(2,targets);
    if (LumenHairProgram) GlDeleteProgram(LumenHairProgram); if (LumenHairComposite) GlDeleteProgram(LumenHairComposite); if (LumenHairUniform) GlDeleteBuffers(1,&LumenHairUniform);
    LumenHairIndirect=LumenHairSamples=LumenHairFramebuffer=LumenHairResolvedFramebuffer=LumenHairProgram=LumenHairComposite=LumenHairUniform=0; LumenHairWidth=LumenHairHeight=0; LumenHairSampleCount=0;
}
bool EnsureLumenHairReceiver()
{
    if (!LumenHairProgram)
    {
        std::string vertex,fragment;
        if (!LoadTechniqueShaderSource("OpenGLES","HairFur.glsl",vertex,LastError) || !LoadTechniqueShaderSource("OpenGLES","LumenHair.glsl",fragment,LastError)) return false;
        vertex.insert(vertex.find('\n')+1,"#define EVANS_VERTEX_STAGE 1\n");
        const auto program=CompileProgramPair(vertex,fragment); if(!program)return false;
        LumenHairProgram=program;
    }
    if (!LumenHairComposite) LumenHairComposite=CompileStageProgram("LumenHair.glsl","#define EVANS_LUMEN_HAIR_COMPOSITE 1\n");
    if (!LumenHairComposite) return false;
    if (!LumenHairUniform) GlGenBuffers(1,&LumenHairUniform);
    if (LumenHairWidth==CreateInfo.Width && LumenHairHeight==CreateInfo.Height && LumenHairSampleCount==FrameSampleCount) return true;
    GLuint texture=0,samples=0,framebuffer=0,resolved=0;
    const auto cleanup=[&]() { const GLuint images[]{texture,samples},targets[]{framebuffer,resolved};GlDeleteTextures(2,images);GlDeleteFramebuffers(2,targets); };
    UnbindSampledTextures(); GlActiveTexture(GL_TEXTURE0); GlBindBuffer(TSRPixelUnpackBuffer,0);
    GlGenTextures(1,&texture); GlBindTexture(GL_TEXTURE_2D,texture); ConfigureDepthTexture();
    GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    GlGenFramebuffers(1,&resolved); GlBindFramebuffer(GL_FRAMEBUFFER,resolved); GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0);
    const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) { cleanup(); return Fail("Fiber irradiance resolve target is incomplete"); }
    GlGenFramebuffers(1,&framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    if (FrameSampleCount>1)
    {
        GlGenTextures(1,&samples); GlBindTexture(GL_TEXTURE_2D_MULTISAMPLE,samples);
        GlTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE,FrameSampleCount,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,GL_TRUE_VALUE);
    }
    const auto kind=FrameSampleCount>1?GL_TEXTURE_2D_MULTISAMPLE:GL_TEXTURE_2D;
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,kind,FrameSampleCount>1?samples:texture,0);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,kind,FrameSampleCount>1?Multisample.Depth:DepthTexture,0); GlDrawBuffers(1,&attachment);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create fiber irradiance targets")) { cleanup(); return false; }
    const GLuint oldImages[]{LumenHairIndirect,LumenHairSamples},oldTargets[]{LumenHairFramebuffer,LumenHairResolvedFramebuffer}; GlDeleteTextures(2,oldImages); GlDeleteFramebuffers(2,oldTargets);
    LumenHairIndirect=texture; LumenHairSamples=samples; LumenHairFramebuffer=framebuffer; LumenHairResolvedFramebuffer=resolved; LumenHairWidth=CreateInfo.Width; LumenHairHeight=CreateInfo.Height; LumenHairSampleCount=FrameSampleCount; return true;
}
bool ResolveLumenHairReceiver()
{
    if (!Hair.LumenReceiver) return true;
    if (!Hair.Resolved || !Hair.HasPrimaryView || !LumenVolumeFrame.Grid.W) return Fail("Fiber irradiance requires the primary surface and current SH volume");
    LumenHairReceiverFrame frame; HairFurConstants geometry;
    if (!BuildLumenHairReceiverFrame(LumenVolumeFrame,Hair.Frame.Settings,Hair.PrimaryView.Lighting,Hair.PrimaryView.ClipPlane,frame,LastError) ||
        !BuildHairFurReplayConstants(Hair.PrimaryView,Hair.Frame,HairFurRasterPhase::Surface,{},geometry,LastError) || !EnsureLumenHairReceiver()) return false;
    UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,LumenHairFramebuffer); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    constexpr GLenum ScissorTest=0x0C11; GlDisable(ScissorTest); GlDisable(GL_BLEND); GlDisable(GL_CULL_FACE); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDisable(GL_SAMPLE_MASK);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlDepthMask(GL_FALSE_VALUE); GlClearColor(0,0,0,0); GlClear(GL_COLOR_BUFFER_BIT); GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_LEQUAL);
    GlBindVertexArray(HairArray); GlBindBuffer(GL_ARRAY_BUFFER,HairVertices); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,HairIndices);
    const std::size_t offsets[]{offsetof(HairVertex,Position),offsetof(HairVertex,Tangent),offsetof(HairVertex,Reflectance),offsetof(HairVertex,Emission),offsetof(HairVertex,Ribbon)};
    for (GLuint attribute=0;attribute<std::size(offsets);++attribute)
    { GlEnableVertexAttribArray(attribute); GlVertexAttribPointer(attribute,attribute<2?3:4,GL_FLOAT,GL_FALSE_VALUE,sizeof(HairVertex),reinterpret_cast<const void*>(offsets[attribute])); }
    GlBindBuffer(GL_UNIFORM_BUFFER,HairUniforms); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(geometry),&geometry); GlBindBufferBase(GL_UNIFORM_BUFFER,0,HairUniforms);
    GlBindBuffer(GL_UNIFORM_BUFFER,LumenHairUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(frame),&frame,GL_DYNAMIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,1,LumenHairUniform);
    GlUseProgram(LumenHairProgram); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,LumenVolumeAtlas); if (LumenBindSampler) LumenBindSampler(0,0);
    GlUniform1i(GlGetUniformLocation(LumenHairProgram,"FiberVolume"),0); GlDrawElements(GL_TRIANGLES,GLsizei(Hair.Frame.Geometry->Indices.size()),GL_UNSIGNED_INT,nullptr);
    GlBindFramebuffer(GL_FRAMEBUFFER,0); UnbindSampledTextures();
    if (FrameSampleCount>1)
    {
        GlBindFramebuffer(GL_READ_FRAMEBUFFER,LumenHairFramebuffer); GlReadBuffer(GL_COLOR_ATTACHMENT0); GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,LumenHairResolvedFramebuffer);
        GlBlitFramebuffer(0,0,CreateInfo.Width,CreateInfo.Height,0,0,CreateInfo.Width,CreateInfo.Height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,LumenGatherFramebuffer); GlDisable(GL_DEPTH_TEST); GlBindVertexArray(VertexArray); GlUseProgram(LumenHairComposite);
    GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,LumenHairIndirect); GlActiveTexture(GL_TEXTURE0+1); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]); if (LumenBindSampler) LumenBindSampler(1,0);
    GlUniform1i(GlGetUniformLocation(LumenHairComposite,"FiberVolume"),0); GlUniform1i(GlGetUniformLocation(LumenHairComposite,"Scene"),1); GlDrawArrays(GL_TRIANGLES,0,3); UnbindSampledTextures();
    GlBindFramebuffer(GL_FRAMEBUFFER,LumenGatherFramebuffer); GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
    GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height); GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,0); GlBindBufferBase(GL_UNIFORM_BUFFER,1,0);
    if (!CheckGL("Resolve fiber irradiance")) return false;
    ++PendingFrame.Statistics.LumenHairReceiverPassCount; ++PendingFrame.Statistics.GlobalIlluminationPassCount; return true;
}
