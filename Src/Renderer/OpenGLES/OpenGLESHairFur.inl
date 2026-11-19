// A private VAO describes world-space ribbon attributes. The framebuffer only
// borrows ordinary HDR, material-normal and depth images; none are sampled here.
HairFurExecution Hair;
HairFurFrame UploadedHair;
std::array<GLuint,HairShadowMapCount> HairShadowTextures{};
bool PrepareHairShadowReceiver(const HairFurDesc& desc,HairFurConstants& constants)
{
    SetHairFurAnimation(Hair.Frame,Hair.Frame.SampleTimeSeconds,constants);
    constants.MotionOptions.Y=Hair.LumenReceiver?1u:0u;
    if (!BuildHairContactControl(desc.ViewProjection,*Hair.Frame.Lights,FrameContact.Enabled && !Hair.Frame.Geometry->Indices.empty(),
        Hair.Contact,ContactGenerated && ContactReceiverActive && ContactShadows.Result(),FrameSampleCount,constants.ContactControl,LastError)) return false;
    HairShadowTextures={}; std::array<HairShadowMapReadiness,HairShadowMapCount> maps{};
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
            if (FrameShadowFilter.Mode==Techniques::ShadowFilterMode::VSM)
            {
                if (!MomentTexture(handle,HairShadowTextures[slot])) return false;
                maps[slot].MomentsReady=true;
            }
        }
    }
    return BuildHairShadowReceiverConstants({desc.LightViewProjection,requested},*Hair.Frame.Lights,FrameCascades,FrameShadowFilter,maps,constants.Shadows,LastError);
}
GLuint HairVertices=0,HairIndices=0,HairLights=0,HairUniforms=0,HairArray=0,HairFramebuffer=0;
std::array<GLuint,5> HairPrograms{};
void DestroyHairFur()
{
    const GLuint buffers[]{HairVertices,HairIndices,HairLights,HairUniforms};
    if (GlDeleteBuffers) GlDeleteBuffers(static_cast<GLsizei>(std::size(buffers)),buffers);
    if (GlDeleteVertexArrays && HairArray) GlDeleteVertexArrays(1,&HairArray);
    if (GlDeleteFramebuffers && HairFramebuffer) GlDeleteFramebuffers(1,&HairFramebuffer);
    if (GlDeleteProgram) for (auto program:HairPrograms) if (program) GlDeleteProgram(program);
    HairVertices=HairIndices=HairLights=HairUniforms=HairArray=HairFramebuffer=0; HairPrograms={}; Hair={}; UploadedHair={};
}
bool EnsureHairFur()
{
    const auto& frame=Hair.Frame;
    if (UploadedHair.Geometry!=frame.Geometry || UploadedHair.Lights!=frame.Lights)
    {
        if (!ValidateHairFurPublication(frame,LastError)) return false;
        GLuint vertices=0,indices=0,lights=0;
        const auto cleanup=[&](){const GLuint buffers[]{vertices,indices,lights}; GlDeleteBuffers(static_cast<GLsizei>(std::size(buffers)),buffers);};
        const auto create=[&](const void* data,std::size_t bytes,GLuint& buffer)
        {
            GlGenBuffers(1,&buffer); GlBindBuffer(GL_ARRAY_BUFFER,buffer);
            GlBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(bytes),data,GL_STATIC_DRAW); GlBindBuffer(GL_ARRAY_BUFFER,0);
            return buffer && CheckGL("Upload hair publication");
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
        { GlDeleteBuffers(1,&HairVertices); GlDeleteBuffers(1,&HairIndices); HairVertices=vertices; HairIndices=indices; }
        if (UploadedHair.Lights!=frame.Lights) { GlDeleteBuffers(1,&HairLights); HairLights=lights; }
        UploadedHair=frame;
    }
    if (!HairUniforms)
    {
        GlGenBuffers(1,&HairUniforms); GlBindBuffer(GL_UNIFORM_BUFFER,HairUniforms);
        GlBufferData(GL_UNIFORM_BUFFER,sizeof(HairFurConstants),nullptr,GL_DYNAMIC_DRAW);
    }
    if (!HairArray) GlGenVertexArrays(1,&HairArray);
    if (!HairFramebuffer) GlGenFramebuffers(1,&HairFramebuffer);
    UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,HairFramebuffer);
    const auto target=FrameSampleCount>1?GL_TEXTURE_2D_MULTISAMPLE:GL_TEXTURE_2D;
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,target,FrameSampleCount>1?Multisample.Textures[SampleColor]:HdrTextures[0],0);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0+1,target,FrameSampleCount>1?Multisample.Textures[SampleMaterialFirst+1]:GBufferTextures[1],0);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,target,FrameSampleCount>1?Multisample.Depth:DepthTexture,0);
    const GLenum colors[]{GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT0+1}; GlDrawBuffers(static_cast<GLsizei>(std::size(colors)),colors);
    const bool complete=GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindBuffer(GL_UNIFORM_BUFFER,0);
    return (complete && CheckGL("Prepare hair raster resources")) || Fail("OpenGL ES hair framebuffer is incomplete");
}
bool DrawHairGeometry(const HairFurConstants& constants,HairFurRasterPhase phase)
{
    if (Hair.Frame.Geometry->Indices.empty()) return true;
    UnbindSampledTextures();
    const auto program=phase==HairFurRasterPhase::Velocity?4:phase==HairFurRasterPhase::Surface?0:phase==HairFurRasterPhase::Capture?3:phase==HairFurRasterPhase::ShadowMoments?2:1;
    // Captures, moments and velocity are optional; compile their programs only
    // when a frame actually uses them.
    if (!HairPrograms[program])
    {
        const char* definitions[]{"","#define EVANS_HAIR_DEPTH_ONLY 1\n","#define EVANS_HAIR_VSM_SHADOW 1\n","#define EVANS_HAIR_CAPTURE 1\n","#define EVANS_HAIR_VELOCITY 1\n"};
        HairPrograms[program]=CompileStageProgram("HairFur.glsl",definitions[program]);
        if (!HairPrograms[program]) return false;
    }
    GlUseProgram(HairPrograms[program]);
    if (phase==HairFurRasterPhase::Surface)
    {
        GlActiveTexture(GL_TEXTURE0+HairShadowMapCount); GlBindTexture(GL_TEXTURE_2D_ARRAY,constants.ContactControl.X?ContactShadows.Result():0);
        GlUniform1i(GlGetUniformLocation(HairPrograms[program],"FiberContactLayers"),HairShadowMapCount);
        const char* names[]{"FiberMap0","FiberMap1","FiberMap2","FiberMap3","FiberMap4"};
        for (GLuint slot=0;slot<HairShadowMapCount;++slot)
        {
            GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,HairShadowTextures[slot]);
            GlUniform1i(GlGetUniformLocation(HairPrograms[program],names[slot]),static_cast<GLint>(slot));
        }
    }
    GlBindVertexArray(HairArray); GlBindBuffer(GL_ARRAY_BUFFER,HairVertices); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,HairIndices);
    const std::size_t offsets[]{offsetof(HairVertex,Position),offsetof(HairVertex,Tangent),offsetof(HairVertex,Reflectance),offsetof(HairVertex,Emission),offsetof(HairVertex,Ribbon)};
    for (GLuint attribute=0;attribute<std::size(offsets);++attribute)
    {
        GlEnableVertexAttribArray(attribute);
        GlVertexAttribPointer(attribute,attribute<2?3:4,GL_FLOAT,GL_FALSE_VALUE,sizeof(HairVertex),reinterpret_cast<const void*>(offsets[attribute]));
    }
    GlBindBuffer(GL_UNIFORM_BUFFER,HairUniforms); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,HairUniforms); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,HairLights);
    GlDisable(GL_BLEND); GlDisable(GL_CULL_FACE); GlDisable(GL_DITHER);
    if (phase==HairFurRasterPhase::ShadowDepth || phase==HairFurRasterPhase::ShadowMoments)
    { GlEnable(GL_POLYGON_OFFSET_FILL); GlPolygonOffset(ShadowSlopeScaledDepthBias,float(ShadowDepthBiasUnits)); }
    else GlDisable(GL_POLYGON_OFFSET_FILL);
    GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_LEQUAL); GlDepthMask(phase==HairFurRasterPhase::Velocity?GL_FALSE_VALUE:GL_TRUE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE);
    const bool perSample=phase==HairFurRasterPhase::Surface && constants.ContactControl.X && FrameSampleCount>1;
    const auto repetitions=perSample?FrameSampleCount:1;
    const GLint sampleLocation=GlGetUniformLocation(HairPrograms[program],"FiberContactSample");
    if (perSample) GlEnable(GL_SAMPLE_MASK); else GlDisable(GL_SAMPLE_MASK);
    // ES 3.1 core keeps each array layer paired with its coverage bit.
    for (GLuint sample=0;sample<repetitions;++sample)
    {
        if (perSample) GlSampleMaski(0,1u<<sample);
        GlUniform1i(sampleLocation,static_cast<GLint>(sample));
        GlDrawElements(GL_TRIANGLES,static_cast<GLsizei>(Hair.Frame.Geometry->Indices.size()),GL_UNSIGNED_INT,nullptr);
    }
    if (perSample) { GlDisable(GL_SAMPLE_MASK); GlSampleMaski(0,std::numeric_limits<GLuint>::max()); }
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0); GlBindBufferBase(GL_UNIFORM_BUFFER,0,0); GlBindVertexArray(VertexArray);
    return CheckGL("Draw native hair ribbons");
}
bool RenderHairFur(const HairFurConstants& constants)
{
    UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,HairFramebuffer); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    const bool result=DrawHairGeometry(constants,HairFurRasterPhase::Surface); GlBindFramebuffer(GL_FRAMEBUFFER,0); return result;
}
bool ReplayHairFur(const HairFurConstants& constants,HairFurRasterPhase phase) { return DrawHairGeometry(constants,phase); }
