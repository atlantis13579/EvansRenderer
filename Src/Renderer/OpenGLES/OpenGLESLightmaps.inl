// Members of OpenGLESRHI::Impl. Keeping this native resource/draw implementation
// beside its API backend avoids exposing GL dispatch details to shared code.
static constexpr std::size_t LightmapFront=0,LightmapBack=1,LightmapCovered=2,LightmapResolved=3,LightmapComposite=4;
std::array<GLuint,5> LightmapTextures{},LightmapFramebuffers{};
GLuint LightmapProgram=0,LightmapVertexArray=0,LightmapVertexBuffer=0;
int LightmapWidth=0,LightmapHeight=0;
std::uint32_t LightmapSamples=0;
std::shared_ptr<const LightmapBakeResult> UploadedLightmap;
LightmapFrame FrameLightmaps;
Techniques::Settings FrameLightmapSettings;
bool LightmapActive=false;

void DestroyLightmapTargets(bool complete=false)
{
    for (std::size_t index=LightmapCovered;index<LightmapTextures.size();++index)
    {
        if (LightmapFramebuffers[index] && GlDeleteFramebuffers) GlDeleteFramebuffers(1,&LightmapFramebuffers[index]);
        if (LightmapTextures[index] && GlDeleteTextures) GlDeleteTextures(1,&LightmapTextures[index]);
        LightmapFramebuffers[index]=LightmapTextures[index]=0;
    }
    LightmapWidth=LightmapHeight=0; LightmapSamples=0;
    if (!complete) return;
    for (auto index:{LightmapFront,LightmapBack})
    { if (LightmapTextures[index] && GlDeleteTextures) GlDeleteTextures(1,&LightmapTextures[index]); LightmapTextures[index]=0; }
    if (LightmapVertexBuffer && GlDeleteBuffers) GlDeleteBuffers(1,&LightmapVertexBuffer);
    if (LightmapVertexArray && GlDeleteVertexArrays) GlDeleteVertexArrays(1,&LightmapVertexArray);
    if (LightmapProgram && GlDeleteProgram) GlDeleteProgram(LightmapProgram);
    LightmapVertexBuffer=LightmapVertexArray=LightmapProgram=0; UploadedLightmap.reset(); FrameLightmaps={}; LightmapActive=false;
}

bool EnsureLightmapResources()
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) return true;
    const auto& source=*FrameLightmaps.Source;
    if (!source.Irradiance || !source.Receivers || !source.Irradiance->Atlas || source.Receivers->Vertices.empty())
        return Fail("OpenGL ES lightmaps require a complete receiver and bake snapshot");
    if (!LightmapProgram) LightmapProgram=CompileStageProgram("Lightmaps.glsl");
    if (!LightmapProgram) return false;
    UnbindSampledTextures(); GlActiveTexture(GL_TEXTURE0);
    if (UploadedLightmap!=FrameLightmaps.Source)
    {
        const auto& bake=*source.Irradiance; const auto side=bake.Atlas->Settings.Resolution;
        if (bake.FrontIrradiance.size()!=static_cast<std::size_t>(side)*side ||
            (!bake.BackIrradiance.empty() && bake.BackIrradiance.size()!=bake.FrontIrradiance.size())) return Fail("OpenGL ES lightmap upload size is invalid");
        for (auto index:{LightmapFront,LightmapBack})
        {
            if (!LightmapTextures[index]) GlGenTextures(1,&LightmapTextures[index]);
            GlBindTexture(GL_TEXTURE_2D,LightmapTextures[index]); ConfigureDepthTexture();
            const auto& image=index==LightmapBack && !bake.BackIrradiance.empty() ? bake.BackIrradiance : bake.FrontIrradiance;
            GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,side,side,0,GL_RGBA,GL_FLOAT,image.data());
        }
        if (!LightmapVertexArray) GlGenVertexArrays(1,&LightmapVertexArray);
        if (!LightmapVertexBuffer) GlGenBuffers(1,&LightmapVertexBuffer);
        GlBindVertexArray(LightmapVertexArray); GlBindBuffer(GL_ARRAY_BUFFER,LightmapVertexBuffer);
        GlBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(source.Receivers->Vertices.size()*sizeof(LightmapReceiverVertex)),
            source.Receivers->Vertices.data(),GL_STATIC_DRAW);
        const GLint components[]{3,2,2,4};
        const std::size_t offsets[]{offsetof(LightmapReceiverVertex,Position),offsetof(LightmapReceiverVertex,MaterialUV),
            offsetof(LightmapReceiverVertex,AtlasUV),offsetof(LightmapReceiverVertex,Color)};
        for (GLuint slot=0;slot<std::size(components);++slot)
        {
            GlEnableVertexAttribArray(slot); GlVertexAttribPointer(slot,components[slot],GL_FLOAT,GL_FALSE_VALUE,
                sizeof(LightmapReceiverVertex),reinterpret_cast<const void*>(offsets[slot]));
        }
        GlBindVertexArray(VertexArray); GlBindTexture(GL_TEXTURE_2D,0);
        if (!CheckGL("Upload lightmap atlases and receiver geometry")) return false;
        UploadedLightmap=FrameLightmaps.Source;
    }
    if (LightmapWidth!=CreateInfo.Width || LightmapHeight!=CreateInfo.Height || LightmapSamples!=FrameSampleCount) DestroyLightmapTargets();
    for (std::size_t index=LightmapCovered;index<LightmapTextures.size();++index)
    {
        const bool multisample=index==LightmapCovered && FrameSampleCount>1;
        const GLenum target=multisample ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
        if (!LightmapTextures[index])
        {
            GlGenTextures(1,&LightmapTextures[index]); GlBindTexture(target,LightmapTextures[index]);
            if (multisample) GlTexStorage2DMultisample(target,FrameSampleCount,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,GL_TRUE_VALUE);
            else
            {
                ConfigureDepthTexture(); GlTexImage2D(target,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
            }
            GlBindTexture(target,0); GlGenFramebuffers(1,&LightmapFramebuffers[index]);
        }
        GlBindFramebuffer(GL_FRAMEBUFFER,LightmapFramebuffers[index]);
        GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,target,LightmapTextures[index],0);
        if (index==LightmapCovered) GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,target,
            multisample ? Multisample.Depth : DepthTexture,0);
        const GLenum attachment=GL_COLOR_ATTACHMENT0; GlDrawBuffers(1,&attachment);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create lightmap framebuffer"))
        { GlBindFramebuffer(GL_FRAMEBUFFER,0); DestroyLightmapTargets(); return Fail("OpenGL ES lightmap framebuffer is incomplete"); }
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,0); LightmapWidth=CreateInfo.Width; LightmapHeight=CreateInfo.Height; LightmapSamples=FrameSampleCount;
    return true;
}

bool ResolveLightmaps(const GlobalIlluminationDesc& desc)
{
    if (!FrameLightmaps.Source || FrameLightmaps.Draws.empty()) { GI.Resolved=true; return true; }
    if (UploadedLightmap!=FrameLightmaps.Source) return Fail("OpenGL ES lightmap snapshot changed during a frame");
    auto restore=[&] { UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindVertexArray(VertexArray); GlDepthMask(GL_TRUE_VALUE); GlDisable(GL_BLEND); };
    UnbindSampledTextures(); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    GlBindFramebuffer(GL_FRAMEBUFFER,LightmapFramebuffers[LightmapCovered]); GlClearColor(0,0,0,0); GlClear(GL_COLOR_BUFFER_BIT);
    GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_EQUAL); GlDepthMask(GL_FALSE_VALUE);
    GlBlendEquation(GL_FUNC_ADD);
    if (FrameLightmaps.KeepFirstSurface) { GlEnable(GL_BLEND); GlBlendFuncSeparate(GL_ONE_MINUS_DST_ALPHA,GL_ONE,GL_ONE,GL_ZERO); }
    else GlDisable(GL_BLEND);
    GlBindVertexArray(LightmapVertexArray); GlUseProgram(LightmapProgram);
    const char* samplerNames[]{"FrontAtlas","BackAtlas","DiffuseMap","OpacityMap","SceneColor","Contribution"};
    auto bind=[&](const std::array<GLuint,6>& textures) {
        for (GLuint slot=0;slot<textures.size();++slot)
        {
            GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,textures[slot]);
            GlUniform1i(GlGetUniformLocation(LightmapProgram,samplerNames[slot]),slot);
        }
    };
    LightmapConstants constants;
    auto upload=[&] {
        const auto world=ToColumnMajor(constants.World), projection=ToColumnMajor(constants.ViewProjection);
        GlUniformMatrix4fv(GlGetUniformLocation(LightmapProgram,"World"),1,GL_FALSE_VALUE,world.data());
        GlUniformMatrix4fv(GlGetUniformLocation(LightmapProgram,"ViewProjection"),1,GL_FALSE_VALUE,projection.data());
        UploadFloat4(GlGetUniformLocation(LightmapProgram,"DiffuseColor"),constants.DiffuseColor);
        UploadFloat4(GlGetUniformLocation(LightmapProgram,"Appearance"),constants.Appearance);
        GlUniform4ui(GlGetUniformLocation(LightmapProgram,"Options"),constants.Options.X,constants.Options.Y,constants.Options.Z,constants.Options.W);
    };
    for (const auto& draw:FrameLightmaps.Draws)
    {
        if (!BuildLightmapConstants(FrameLightmaps,draw,desc.ViewProjection,FrameGI.Intensity,FrameLightmapSettings.Shading,FrameLightmapSettings.Metallic,constants))
        { restore(); return Fail("OpenGL ES lightmap receiver range or transform is invalid"); }
        auto texture=[&](TextureHandle handle) -> GLuint {
            const auto found=Textures.find(handle.Value);
            return handle && found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? found->second.Texture : 0;
        };
        const auto diffuse=texture(draw.DiffuseTexture),opacity=texture(draw.OpacityTexture);
        if ((draw.DiffuseTexture && !diffuse) || (draw.OpacityTexture && !opacity)) { restore(); return Fail("OpenGL ES lightmap material texture is invalid"); }
        upload(); bind({LightmapTextures[LightmapFront],LightmapTextures[LightmapBack],diffuse,opacity,0,0});
        if (draw.BackFaceCulling) { GlEnable(GL_CULL_FACE); GlCullFace(GL_BACK); } else GlDisable(GL_CULL_FACE);
        GlFrontFace(RealtimeWorldIsMirrored(draw.World) ? GL_CW : GL_CCW);
        GlDrawArrays(GL_TRIANGLES,draw.FirstVertex,draw.VertexCount);
    }
    UnbindSampledTextures(); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE); GlDisable(GL_BLEND);
    GlBindFramebuffer(GL_READ_FRAMEBUFFER,LightmapFramebuffers[LightmapCovered]); GlReadBuffer(GL_COLOR_ATTACHMENT0);
    GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,LightmapFramebuffers[LightmapResolved]);
    GlBlitFramebuffer(0,0,CreateInfo.Width,CreateInfo.Height,0,0,CreateInfo.Width,CreateInfo.Height,GL_COLOR_BUFFER_BIT,GL_NEAREST);
    GlBindFramebuffer(GL_FRAMEBUFFER,LightmapFramebuffers[LightmapComposite]);
    constants.Options.W=static_cast<unsigned>(LightmapPass::Composite);
    constants.Options.X=AmbientOcclusionResolved ? 1u : 0u; upload();
    bind({AmbientOcclusionResolved ? AmbientOcclusionTextures[0] : 0,0,0,0,HdrTextures[0],LightmapTextures[LightmapResolved]}); GlDrawArrays(GL_TRIANGLES,0,3);
    UnbindSampledTextures(); GlReadBuffer(GL_COLOR_ATTACHMENT0); GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]);
    GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
    const bool valid=CheckGL("Resolve lightmap surface illumination"); restore();
    if (!valid) return false;
    GI.Resolved=true; PendingFrame.Statistics.GlobalIlluminationPassCount=2;
    RecordLightmapDraws(FrameLightmaps,PendingFrame.Statistics); return true;
}
