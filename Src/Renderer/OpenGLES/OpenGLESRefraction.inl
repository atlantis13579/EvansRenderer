// OpenGLESRHI::Impl members. Private MRT/depth capture and a separate HDR output
// keep sampled textures detached from the active framebuffer throughout the pass.
std::array<GLuint,4> RefractionTextures{}; // Normal, tint, composition, depth.
std::array<GLuint,2> RefractionFramebuffers{},RefractionPrograms{};
GLuint RefractionVAO=0,RefractionVertices=0,RefractionIndices=0;
int RefractionWidth=0,RefractionHeight=0;
RefractionFrame FrameRefraction;
Techniques::RefractionSettings FrameRefractionSettings;
std::shared_ptr<const RefractionGeometry> UploadedRefraction;
bool RefractionActive=false,RefractionResolved=false;

void DestroyRefractionTargets(bool complete=false)
{
    for (auto& framebuffer:RefractionFramebuffers) { if (framebuffer && GlDeleteFramebuffers) GlDeleteFramebuffers(1,&framebuffer); framebuffer=0; }
    for (auto& texture:RefractionTextures) { if (texture && GlDeleteTextures) GlDeleteTextures(1,&texture); texture=0; }
    RefractionWidth=RefractionHeight=0;
    if (!complete) return;
    for (auto& program:RefractionPrograms) { if (program && GlDeleteProgram) GlDeleteProgram(program); program=0; }
    for (auto* buffer:{&RefractionVertices,&RefractionIndices}) { if (*buffer && GlDeleteBuffers) GlDeleteBuffers(1,buffer); *buffer=0; }
    if (RefractionVAO && GlDeleteVertexArrays) GlDeleteVertexArrays(1,&RefractionVAO);
    RefractionVAO=0; UploadedRefraction.reset(); FrameRefraction={}; RefractionActive=RefractionResolved=false;
}

bool EnsureRefractionResources()
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) return true;
    if (!RefractionPrograms[0]) RefractionPrograms[0]=CompileStageProgram("Refraction.glsl","#define EVANS_REFRACTION_CAPTURE 1\n");
    if (!RefractionPrograms[1]) RefractionPrograms[1]=CompileStageProgram("Refraction.glsl");
    if (!RefractionPrograms[0] || !RefractionPrograms[1]) return false;
    if (UploadedRefraction!=FrameRefraction.Geometry)
    {
        const auto& geometry=*FrameRefraction.Geometry;
        if (geometry.Vertices.empty() || geometry.Indices.empty()) return Fail("OpenGL ES refraction geometry is empty");
        if (!RefractionVAO) GlGenVertexArrays(1,&RefractionVAO);
        if (!RefractionVertices) GlGenBuffers(1,&RefractionVertices);
        if (!RefractionIndices) GlGenBuffers(1,&RefractionIndices);
        GlBindVertexArray(RefractionVAO); GlBindBuffer(GL_ARRAY_BUFFER,RefractionVertices);
        GlBufferData(GL_ARRAY_BUFFER,geometry.Vertices.size()*sizeof(Vertex),geometry.Vertices.data(),GL_STATIC_DRAW);
        GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,RefractionIndices);
        GlBufferData(GL_ELEMENT_ARRAY_BUFFER,geometry.Indices.size()*sizeof(std::uint32_t),geometry.Indices.data(),GL_STATIC_DRAW);
        const GLint components[]{3,3,4,2};
        const std::size_t offsets[]{offsetof(Vertex,Position),offsetof(Vertex,Normal),offsetof(Vertex,Color),offsetof(Vertex,TexCoord)};
        for (GLuint slot=0;slot<std::size(components);++slot)
        { GlEnableVertexAttribArray(slot); GlVertexAttribPointer(slot,components[slot],GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsets[slot])); }
        GlBindVertexArray(VertexArray); UploadedRefraction.reset();
        if (!CheckGL("Upload refraction geometry")) return false;
        UploadedRefraction=FrameRefraction.Geometry;
    }
    if (RefractionWidth!=CreateInfo.Width || RefractionHeight!=CreateInfo.Height) DestroyRefractionTargets();
    if (RefractionFramebuffers[0] && RefractionFramebuffers[1]) return true;
    for (std::size_t index=0;index<RefractionTextures.size();++index)
    {
        GlGenTextures(1,&RefractionTextures[index]); GlBindTexture(GL_TEXTURE_2D,RefractionTextures[index]); ConfigureDepthTexture();
        if (index==3) GlTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,CreateInfo.Width,CreateInfo.Height,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
        else GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,CreateInfo.Width,CreateInfo.Height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
    }
    GlBindTexture(GL_TEXTURE_2D,0); GlGenFramebuffers(2,RefractionFramebuffers.data());
    GlBindFramebuffer(GL_FRAMEBUFFER,RefractionFramebuffers[0]);
    const GLenum attachments[]{GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT0+1};
    for (unsigned index=0;index<2;++index) GlFramebufferTexture2D(GL_FRAMEBUFFER,attachments[index],GL_TEXTURE_2D,RefractionTextures[index],0);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,RefractionTextures[3],0); GlDrawBuffers(2,attachments);
    bool valid=GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    GlBindFramebuffer(GL_FRAMEBUFFER,RefractionFramebuffers[1]);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,RefractionTextures[2],0); GlDrawBuffers(1,attachments);
    valid=valid && GlCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE;
    GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!CheckGL("Create refraction targets") || !valid) { DestroyRefractionTargets(); return Fail("OpenGL ES refraction framebuffer is incomplete"); }
    RefractionWidth=CreateInfo.Width; RefractionHeight=CreateInfo.Height; return true;
}

bool ResolveRefraction(const RefractionDesc& desc)
{
    if (!FrameRefraction.Geometry || FrameRefraction.Draws.empty()) { RefractionResolved=true; return true; }
    if (!EnsureRefractionResources()) return false;
    RefractionConstants base;
    if (!BuildRefractionConstants(desc.ViewProjection,desc.Environment,FrameRefractionSettings,CreateInfo.Width,CreateInfo.Height,base)) return Fail("OpenGL ES refraction projection is invalid");
    auto restore=[&] { UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindVertexArray(VertexArray); GlDepthMask(GL_TRUE_VALUE); GlDisable(GL_BLEND); };
    auto upload=[&](GLuint program,const RefractionConstants& value) {
        const Matrix4* matrices[]{&value.World,&value.NormalWorld,&value.ViewProjection,&value.InverseViewProjection};
        const char* names[]{"World","NormalWorld","ViewProjection","InverseViewProjection"};
        for (unsigned index=0;index<std::size(matrices);++index) { const auto matrix=ToColumnMajor(*matrices[index]); GlUniformMatrix4fv(GlGetUniformLocation(program,names[index]),1,GL_FALSE_VALUE,matrix.data()); }
        const Float4* vectors[]{&value.Color,&value.Material,&value.Optics,&value.Absorption,&value.Viewport,&value.Trace,&value.Appearance,&value.Environment};
        const char* vectorNames[]{"Color","Material","Optics","Absorption","Viewport","Trace","Appearance","Environment"};
        for (unsigned index=0;index<std::size(vectors);++index) UploadFloat4(GlGetUniformLocation(program,vectorNames[index]),*vectors[index]);
        GlUniform4ui(GlGetUniformLocation(program,"Textures"),value.Textures.X,value.Textures.Y,value.Textures.Z,value.Textures.W);
        GlUniform4ui(GlGetUniformLocation(program,"Options"),value.Options.X,value.Options.Y,value.Options.Z,value.Options.W);
    };
    auto bind=[&](GLuint program,const std::array<GLuint,8>& images) {
        const char* names[]{"OpaqueDepth","DiffuseMap","NormalMap","OpacityMap","SceneColor","SurfaceDepth","SurfaceNormal","SurfaceTint"};
        for (GLuint slot=0;slot<images.size();++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,images[slot]); GlUniform1i(GlGetUniformLocation(program,names[slot]),slot); }
    };
    auto texture=[&](TextureHandle handle) -> GLuint {
        const auto found=Textures.find(handle.Value);
        return handle && found!=Textures.end() && found->second.Description.Format==TextureFormat::RGBA8 ? found->second.Texture : 0;
    };
    UnbindSampledTextures(); GlDisable(GL_BLEND); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL);
    GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); GlViewport(0,0,CreateInfo.Width,CreateInfo.Height);
    GlBindFramebuffer(GL_FRAMEBUFFER,RefractionFramebuffers[0]); GlDepthMask(GL_TRUE_VALUE); GlDepthFunc(GL_LESS); GlEnable(GL_DEPTH_TEST);
    GlClearColor(0,0,0,0); GlClearDepthf(1); GlClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    GlUseProgram(RefractionPrograms[0]); GlBindVertexArray(RefractionVAO);
    for (const auto& draw:FrameRefraction.Draws)
    {
        RefractionConstants value; RefractionGeometryRange range;
        if (!BuildRefractionDrawConstants(FrameRefraction,draw,base,value,range)) { restore(); return Fail("OpenGL ES refraction draw is invalid"); }
        if (!range.IndexCount) continue;
        const auto diffuse=texture(draw.DiffuseTexture),normal=texture(draw.NormalTexture),opacity=texture(draw.OpacityTexture);
        if ((draw.DiffuseTexture && !diffuse) || (draw.NormalTexture && !normal) || (draw.OpacityTexture && !opacity)) { restore(); return Fail("OpenGL ES refraction material texture is invalid"); }
        upload(RefractionPrograms[0],value); bind(RefractionPrograms[0],{DepthTexture,diffuse,normal,opacity,0,0,0,0});
        if (draw.BackFaceCulling) { GlEnable(GL_CULL_FACE); GlCullFace(GL_BACK); } else GlDisable(GL_CULL_FACE);
        GlFrontFace(RealtimeWorldIsMirrored(draw.World) ? GL_CW : GL_CCW);
        GlDrawElements(GL_TRIANGLES,range.IndexCount,GL_UNSIGNED_INT,reinterpret_cast<const void*>(std::size_t(range.FirstIndex)*sizeof(std::uint32_t)));
    }
    UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER,RefractionFramebuffers[1]); GlDisable(GL_DEPTH_TEST); GlDisable(GL_CULL_FACE);
    GlUseProgram(RefractionPrograms[1]); GlBindVertexArray(VertexArray); upload(RefractionPrograms[1],base);
    BindLumenTranslucency(RefractionPrograms[1],true);
    bind(RefractionPrograms[1],{DepthTexture,0,0,0,HdrTextures[0],RefractionTextures[3],RefractionTextures[0],RefractionTextures[1]});
    GlDrawArrays(GL_TRIANGLES,0,3); UnbindSampledTextures(); GlReadBuffer(GL_COLOR_ATTACHMENT0);
    GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,HdrTextures[0]); GlCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,CreateInfo.Width,CreateInfo.Height);
    const bool valid=CheckGL("Resolve screen-space refraction"); restore(); if (!valid) return false;
    RefractionResolved=true; RecordRefractionDraws(FrameRefraction,PendingFrame.Statistics); return true;
}
