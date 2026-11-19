// RGBA32F needs sampling only; filtering/reconstruction is explicit in GLSL.
// The volume is uploaded before recording and never aliases a color attachment.
VoxelGIFrame FrameVoxelGI;
std::shared_ptr<const VoxelRadianceVolume> UploadedVoxelVolume;
GLuint VoxelVolume=0,VoxelGIProgram=0,VoxelGIUniforms=0;
std::array<GLuint,2> VoxelGIImages{},VoxelGIFramebuffers{};
int VoxelGIWidth=0,VoxelGIHeight=0;
bool VoxelSamplingSupported=false;
bool ProbeVoxelSamplingSupport()
{
    const auto* bytes=GlGetString(GL_VERSION); const std::string version=bytes ? reinterpret_cast<const char*>(bytes):"";
    if ((version.find("OpenGL ES 3.1")==std::string::npos && version.find("OpenGL ES 3.2")==std::string::npos) ||
        MaximumFragmentTextures<7 || MaximumTextureSize<2048 || MaximumUniformBlockSize<static_cast<GLint>(sizeof(VoxelGIConstants))) return false;
    GLuint image=0; GlGenTextures(1,&image); GlBindTexture(GL_TEXTURE_2D,image); ConfigureDepthTexture();
    GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,1,1,0,GL_RGBA,GL_FLOAT,nullptr);
    const bool supported=CheckGL("Probe sampled voxel radiance"); GlBindTexture(GL_TEXTURE_2D,0); GlDeleteTextures(1,&image); return supported;
}
void DestroyVoxelGITargets()
{
    if (GlDeleteFramebuffers) GlDeleteFramebuffers(2,VoxelGIFramebuffers.data());
    if (GlDeleteTextures) GlDeleteTextures(2,VoxelGIImages.data());
    VoxelGIFramebuffers={}; VoxelGIImages={}; VoxelGIWidth=VoxelGIHeight=0;
}
void DestroyVoxelGI()
{
    DestroyVoxelGITargets();
    if (GlDeleteTextures && VoxelVolume) GlDeleteTextures(1,&VoxelVolume);
    if (GlDeleteProgram && VoxelGIProgram) GlDeleteProgram(VoxelGIProgram);
    if (GlDeleteBuffers && VoxelGIUniforms) GlDeleteBuffers(1,&VoxelGIUniforms);
    VoxelVolume=VoxelGIProgram=VoxelGIUniforms=0; UploadedVoxelVolume.reset(); FrameVoxelGI={};
}
bool EnsureVoxelGIResources()
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants({},frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("OpenGL ES voxel GI publication is invalid");
    if (frame.Volume!=UploadedVoxelVolume)
    {
        GLint limit=0; GlGetIntegerv(GL_MAX_TEXTURE_SIZE,&limit); const auto& volume=*frame.Volume;
        if (volume.Width>static_cast<unsigned>(limit) || volume.Height>static_cast<unsigned>(limit)) return Fail("Voxel radiance exceeds ES texture dimensions");
        GLuint image=0; UnbindSampledTextures(); GlGenTextures(1,&image); GlBindTexture(GL_TEXTURE_2D,image); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,volume.Width,volume.Height,0,GL_RGBA,GL_FLOAT,volume.Texels.data());
        GlBindTexture(GL_TEXTURE_2D,0);
        if (!CheckGL("Upload voxel radiance")) { GlDeleteTextures(1,&image); return false; }
        if (VoxelVolume) GlDeleteTextures(1,&VoxelVolume);
        VoxelVolume=image; UploadedVoxelVolume=frame.Volume;
    }
    if (!VoxelGIProgram) VoxelGIProgram=CompileStageProgram("VoxelGlobalIllumination.glsl");
    if (!VoxelGIProgram) return false;
    if (!VoxelGIUniforms)
    {
        GlGenBuffers(1,&VoxelGIUniforms); GlBindBuffer(GL_UNIFORM_BUFFER,VoxelGIUniforms);
        GlBufferData(GL_UNIFORM_BUFFER,sizeof(VoxelGIConstants),nullptr,GL_DYNAMIC_DRAW);
    }
    if (VoxelGIWidth==CreateInfo.Width && VoxelGIHeight==CreateInfo.Height) return true;
    DestroyVoxelGITargets(); UnbindSampledTextures(); GlGenTextures(2,VoxelGIImages.data()); GlGenFramebuffers(2,VoxelGIFramebuffers.data());
    bool complete=true;
    for (std::size_t index=0;index<VoxelGIImages.size();++index)
    {
        GlBindTexture(GL_TEXTURE_2D,VoxelGIImages[index]); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
        GlBindFramebuffer(GL_FRAMEBUFFER,VoxelGIFramebuffers[index]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,VoxelGIImages[index],0);
        const GLenum color=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&color);
        complete=complete && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    }
    GlBindTexture(GL_TEXTURE_2D,0); GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!complete || !CheckGL("Create voxel GI targets")) { DestroyVoxelGITargets(); return Fail("OpenGL ES voxel GI framebuffer is incomplete"); }
    VoxelGIWidth=CreateInfo.Width; VoxelGIHeight=CreateInfo.Height; return true;
}
bool ApplyVoxelGI(const GlobalIlluminationDesc& desc)
{
    const auto& frame=FrameVoxelGI; VoxelGIConstants constants;
    if (!frame.Volume || !BuildVoxelGIConstants(desc.ViewProjection,frame.Settings,*frame.Volume,frame.Intensity,frame.MaximumDistance,frame.Shading,
        CreateInfo.Width,CreateInfo.Height,Techniques::VoxelGIPass::ConeTrace,constants)) return Fail("OpenGL ES voxel GI camera is invalid");
    const auto restore=[&]
    {
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindBufferBase(GL_UNIFORM_BUFFER,0,0);
        GlDisable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD); GlBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        GlDepthMask(GL_TRUE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    };
    UnbindSampledTextures(); GlUseProgram(VoxelGIProgram); GlBindVertexArray(VertexArray);
    constexpr std::array<const char*,7> samplers{"CameraDepth","NormalMaterial","DiffuseMaterial","AmbientMaterial","DirectionalRadiance","PreviousSignal","SceneRadiance"};
    for (std::size_t slot=0;slot<samplers.size();++slot) GlUniform1i(GlGetUniformLocation(VoxelGIProgram,samplers[slot]),static_cast<GLint>(slot));
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,VoxelGIUniforms);
    GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDepthMask(GL_FALSE_VALUE);
    for (const auto pass:VoxelGIPasses)
    {
        const auto index=static_cast<std::size_t>(pass); const bool composite=pass==Techniques::VoxelGIPass::Composite;
        UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,composite ? LightingFramebuffer:VoxelGIFramebuffers[index%2]);
        GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,composite ? GL_FALSE_VALUE:GL_TRUE_VALUE);
        if (composite) { GlEnable(GL_BLEND); GlBlendEquation(GL_FUNC_ADD); GlBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE); }
        else GlDisable(GL_BLEND);
        constants.Options.X=static_cast<unsigned>(pass); GlBindBuffer(GL_UNIFORM_BUFFER,VoxelGIUniforms); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants);
        const auto previous=index ? VoxelGIImages[(index-1)%2]:0;
        const std::array<GLuint,7> inputs=composite ? std::array<GLuint,7>{0,0,0,0,0,previous,0} :
            std::array<GLuint,7>{DepthTexture,GBufferTextures[1],GBufferTextures[0],GBufferTextures[3],VoxelVolume,previous,HdrTextures[0]};
        for (std::size_t slot=0;slot<inputs.size();++slot)
        { GlActiveTexture(GL_TEXTURE0+static_cast<GLenum>(slot)); GlBindTexture(GL_TEXTURE_2D,inputs[slot]); }
        GlDrawArrays(GL_TRIANGLES,0,3);
        if (!CheckGL("Draw voxel GI pass")) { restore(); return false; }
        PendingFrame.VoxelGIPasses.push_back(pass); ++PendingFrame.Statistics.GlobalIlluminationPassCount;
    }
    restore(); GI.Resolved=true; return true;
}
