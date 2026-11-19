// ES storage barriers connect vertex pulling, hardware visibility and the two
// atomic compute phases. Canonical top-left pixels are converted in the shader.
NaniteFrameRasterWork NaniteRasterPlan;
std::shared_ptr<NaniteFrameBuffer> NaniteRasterDecisions;
NaniteDepthHandle NaniteRasterMainDepth;
NaniteVisibilityHandle NaniteRasterPost;
bool PrepareNaniteRasterVisibility(const VisibilityBufferDesc& frame)
{
    // Previous frame resources are retired only after the frame fence/abort.
    NaniteDepths.erase(NaniteRasterMainDepth.Value); NaniteDecisions.erase(NaniteRasterPost.Value);
    NaniteRasterMainDepth={}; NaniteRasterPost={}; NaniteRasterDecisions.reset();
    if (!NaniteFrameDummy) { const unsigned zero=0; if (!CreateNaniteFrameBuffer(1,&zero,false,NaniteFrameDummy)) return false; }
    NaniteRasterDecisions=NaniteFrameDummy;
    if (!frame.NaniteVisibility) return true;
    const auto found=NaniteDecisions.find(frame.NaniteVisibility.Value);
    if (found==NaniteDecisions.end() || found->second->Info.Published || found->second->Info.Phase!=NaniteVisibilityPhase::Main ||
        found->second->Info.Frame!=FrameNanite.CurrentFrame() || found->second->Info.QueryCount!=frame.NaniteClusters.size() || found->second->Draws)
        return Fail("Nanite raster requires matching unconsumed main-phase cluster decisions");
    NaniteRasterDecisions=found->second->Data; return true;
}
bool RetestNaniteRaster(const VisibilityBufferDesc& frame)
{
    if (!CaptureNaniteDepth(NaniteRasterMainDepth,true) || !RetestNaniteVisibility(frame.NaniteVisibility,NaniteRasterMainDepth,NaniteRasterPost,true)) return false;
    NaniteRasterDecisions=NaniteDecisions.at(NaniteRasterPost.Value)->Data;
    for (auto& job:NaniteRasterPlan.Dispatches) { job.Constants.Raster.Quality.Z=unsigned(NaniteVisibilityState::PostVisible); job.Constants.Raster.Quality.W|=NaniteRasterPreserveMain; }
    return true;
}
std::shared_ptr<NaniteFrameBuffer> NaniteRasterInput,NaniteRasterOutput;
std::array<GLuint,6> NaniteRasterFramePrograms{};
std::array<GLuint,2> NaniteRasterGraphics{};
GLuint NaniteRasterUniform=0;
void DestroyNaniteFrameRaster()
{
    DestroyNaniteHardware();
    NaniteRasterDecisions.reset(); NaniteRasterMainDepth={}; NaniteRasterPost={};
    NaniteRasterInput.reset(); NaniteRasterOutput.reset(); NaniteRasterPlan={};
    for (auto& program:NaniteRasterFramePrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; }
    for (auto& program:NaniteRasterGraphics) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; }
    if (NaniteRasterUniform && GlDeleteBuffers) GlDeleteBuffers(1,&NaniteRasterUniform); NaniteRasterUniform=0;
}
bool PrepareNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (!PrepareNaniteRasterVisibility(frame)) return false;
    if (!NaniteGeometrySupported || !LoadGL(NaniteBindSampler,"glBindSampler")) return Fail("ES hybrid raster requires compute and sampler bindings");
    GLint vertexBlocks=0,fragmentBlocks=0; GlGetIntegerv(VisibilityMaxVertexBlocks,&vertexBlocks); GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&fragmentBlocks);
    if (vertexBlocks<3 || fragmentBlocks<2) return Fail("ES hybrid raster requires three vertex and two fragment storage blocks");
    NaniteFrameRasterWork plan;
    if (!BuildNaniteFrameRasterWork(frame,CreateInfo.Width,CreateInfo.Height,plan,LastError)) return false;
    for (const auto& job:plan.Dispatches)
        if (job.GroupsX>unsigned(NaniteMaximumGroupsX) || job.GroupsY>unsigned(NaniteMaximumGroupsY)) return Fail("ES hybrid raster dispatch exceeds device limits");
    if (!CreateNaniteFrameBuffer(plan.InputWords,nullptr,true,NaniteRasterInput) || !CreateNaniteFrameBuffer(plan.OutputWords,nullptr,true,NaniteRasterOutput)) return false;
    if (!NaniteRasterUniform) GlGenBuffers(1,&NaniteRasterUniform);
    GLint previous=0; GlGetIntegerv(GpuUniformBinding,&previous); GlBindBuffer(GL_UNIFORM_BUFFER,NaniteRasterUniform);
    GlBufferData(GL_UNIFORM_BUFFER,sizeof(NaniteFrameRasterConstants),nullptr,GL_DYNAMIC_DRAW); GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(previous));
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","NaniteRaster.glsl",source,LastError)) return false;
    const auto versionEnd=source.find('\n'); if (versionEnd==std::string::npos) return Fail("ES Nanite raster shader has no version");
    for (unsigned pass=0;pass<NaniteRasterFramePrograms.size();++pass) if (!NaniteRasterFramePrograms[pass])
    {
        auto code=source; code.insert(versionEnd+1,VisibilityShaderDefines()+"#define "+(pass<3?"EVANS_NANITE_RASTER_PASS ":"EVANS_NANITE_FRAME_PASS ")+std::to_string(pass)+"\n");
        const auto program=CompileComputeProgram(code); if(!program)return false;
        NaniteRasterFramePrograms[pass]=program;
    }
    if (!NaniteRasterGraphics[0]) NaniteRasterGraphics[0]=CompileStageProgram("NaniteRaster.glsl",(VisibilityShaderDefines()+"#define EVANS_NANITE_GRAPHICS 1\n").c_str());
    if (!NaniteRasterGraphics[1]) NaniteRasterGraphics[1]=CompileStageProgram("NaniteRaster.glsl",(VisibilityShaderDefines()+"#define EVANS_NANITE_GRAPHICS 1\n#define EVANS_NANITE_EXPORT 1\n").c_str());
    if (!NaniteRasterGraphics[0] || !NaniteRasterGraphics[1]) return false;
    NaniteRasterPlan=std::move(plan); return CheckGL("Prepare Nanite frame raster");
}
bool DispatchNaniteFrameRaster(const NaniteFrameRasterDispatch& job)
{
    GlBindFramebuffer(GL_FRAMEBUFFER,0); const auto program=NaniteRasterFramePrograms[job.Pass]; GlUseProgram(program);
    GlBindBuffer(GL_UNIFORM_BUFFER,NaniteRasterUniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(job.Constants),&job.Constants); GlBindBufferBase(GL_UNIFORM_BUFFER,0,NaniteRasterUniform);
    const bool core=job.Pass<3;
    if (core) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,NaniteRasterInput->Buffer); else BindVisibilityGeometry(0);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,core?NaniteRasterOutput->Buffer:NaniteRasterInput->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,job.Pass==4 && NaniteRasterDecisions?NaniteRasterDecisions->Buffer:0);
    if (job.Pass==5)
    {
        GpuMemoryBarrier(NaniteFramebufferBarrier|NaniteTextureFetchBarrier);
        const GLuint images[]{DepthTexture,VisibilityImages[0]}; const char* names[]{"RasterDepth","RasterVisibility"};
        for (GLuint i=0;i<2;++i)
        { GlActiveTexture(GL_TEXTURE0+i); NaniteBindSampler(i,0); GlBindTexture(GL_TEXTURE_2D,images[i]); GlUniform1i(GlGetUniformLocation(program,names[i]),i); }
    }
    GpuDispatch(job.GroupsX,job.GroupsY,1); GpuMemoryBarrier(GpuStorageBarrier);
    if (job.Pass==5) for (GLuint i=0;i<2;++i) { GlActiveTexture(GL_TEXTURE0+i); GlBindTexture(GL_TEXTURE_2D,0); }
    ++PendingFrame.Statistics.NaniteRasterDispatchCount; return CheckGL("Dispatch Nanite frame raster");
}
bool DrawNaniteRasterPhase(const VisibilityBufferDesc& frame)
{
    if (frame.NaniteVisibility) ++PendingFrame.Statistics.NaniteOcclusionPhases;
    NaniteFrameBindings restore(*this);
    GLint previousTexture=0,previousSampler=0; GlActiveTexture(GL_TEXTURE0+1); GlGetIntegerv(NaniteTextureBinding,&previousTexture); GlGetIntegerv(NaniteSamplerBinding,&previousSampler);
    const auto restoreSecond=[&] { GlActiveTexture(GL_TEXTURE0+1); GlBindTexture(GL_TEXTURE_2D,GLuint(previousTexture)); NaniteBindSampler(1,GLuint(previousSampler)); };
    struct RestoreSecond { const decltype(restoreSecond)& Function; ~RestoreSecond() { Function(); } } second{restoreSecond};
    const auto bind=[&](const VisibilityMaterialDraw* draw,bool exportPass)
    {
        NaniteRasterDrawConstants constants; if (!BuildNaniteRasterDrawConstants(frame,draw,CreateInfo.Width,CreateInfo.Height,constants)) return false;
        if (NaniteRasterPost) constants.Raster.Quality.W|=NaniteRasterPreserveMain;
        const auto program=NaniteRasterGraphics[exportPass?1:0]; GlUseProgram(program);
        GlBindBuffer(GL_UNIFORM_BUFFER,NaniteRasterUniform); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants); GlBindBufferBase(GL_UNIFORM_BUFFER,0,NaniteRasterUniform);
        const GLuint buffers[]{VisibilityGeometry,NaniteRasterInput->Buffer,NaniteRasterOutput->Buffer};
        for (GLuint i=1;i<3;++i) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,i,buffers[i]);
        BindVisibilityGeometry(0);
        const TextureHandle handles[]{draw?draw->Pipeline.DiffuseTexture:TextureHandle{},draw?draw->Pipeline.OpacityTexture:TextureHandle{}};
        const char* names[]{"RasterDiffuse","RasterOpacity"};
        for (GLuint i=0;i<2;++i)
        {
            GLuint texture=0;
            if (handles[i])
            { const auto found=Textures.find(handles[i].Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Invalid ES Nanite material texture"); texture=found->second.Texture; }
            GlActiveTexture(GL_TEXTURE0+i); NaniteBindSampler(i,0); GlBindTexture(GL_TEXTURE_2D,texture); GlUniform1i(GlGetUniformLocation(program,names[i]),i);
        }
        return true;
    };
    const auto targets=[&](bool exportPass)
    {
        GlBindFramebuffer(GL_FRAMEBUFFER,VisibilityFramebuffer); GlBindVertexArray(VisibilityVertexArray);
        GlViewport(0,0,CreateInfo.Width,CreateInfo.Height); GlDisable(GL_BLEND); GlEnable(GL_DEPTH_TEST); GlDepthMask(GL_TRUE_VALUE);
        GlDepthFunc(exportPass?GL_ALWAYS:NaniteRasterPost?GL_LEQUAL:GL_LESS);
    };
    for (unsigned i=0;i<NaniteRasterPlan.HardwareBoundary;++i) if (!DispatchNaniteFrameRaster(NaniteRasterPlan.Dispatches[i])) return false;
    targets(false);
    if (frame.GpuHardware)
    { if (!DrawNaniteHardware(frame)) return false; }
    else
    {
        for (std::size_t order=0;order<frame.Draws.size();++order)
        {
            const auto& draw=frame.Draws[NaniteRasterPost?frame.Draws.size()-1-order:order];
            const auto& range=frame.Scene->Draws[draw.Range]; if (!bind(&draw,false)) return false;
            if (draw.Pipeline.BackFaceCulling) GlEnable(GL_CULL_FACE); else GlDisable(GL_CULL_FACE);
            GlCullFace(GL_BACK); GlFrontFace(range.Mirrored?GL_CW:GL_CCW); GlDrawArrays(GL_TRIANGLES,0,GLsizei(range.TriangleCount*VisibilityTriangleCorners));
        }
    }
    for (std::size_t i=NaniteRasterPlan.HardwareBoundary;i<NaniteRasterPlan.Dispatches.size();++i) if (!DispatchNaniteFrameRaster(NaniteRasterPlan.Dispatches[i])) return false;
    targets(true); GlDisable(GL_CULL_FACE); if (!bind(nullptr,true)) return false; GlDrawArrays(GL_TRIANGLES,0,3); ++PendingFrame.Statistics.NaniteRasterExportCount;
    return CheckGL("Merge Nanite coverage into visibility targets");
}

bool DrawNaniteFrameRaster(const VisibilityBufferDesc& frame)
{
    if (frame.Scene->HasNativeGeometry() && !(frame.GpuInstances?DrawNaniteInstances():DrawNaniteVisibilityGeometry())) return false;
    PendingFrame.Statistics.NaniteOcclusionQueries+=unsigned(frame.NaniteClusters.size());
    if (!DrawNaniteRasterPhase(frame)) return false;
    if (frame.NaniteVisibility)
    {
        if (!RetestNaniteRaster(frame)) return false;
        if (!DrawNaniteRasterPhase(frame)) return false;
    }
    return true;
}
