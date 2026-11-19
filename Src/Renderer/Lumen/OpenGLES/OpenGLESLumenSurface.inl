struct LumenSurfaceRecord
{
    Impl& Owner; LumenSurfaceIdentity Identity; LumenSurfaceLayout Layout;
    std::vector<LumenSurfaceCapture> Captures;
    std::array<GLuint,LumenSurfacePlanes> Images{}; GLuint Depth=0,Framebuffer=0;
    explicit LumenSurfaceRecord(Impl& owner):Owner(owner) {}
    ~LumenSurfaceRecord()
    { if (Framebuffer) Owner.GlDeleteFramebuffers(1,&Framebuffer); if (Depth) Owner.GlDeleteTextures(1,&Depth); Owner.GlDeleteTextures(LumenSurfacePlanes,Images.data()); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenSurfaceRecord>> LumenSurfaces;
std::array<GLuint,2> LumenSurfacePrograms{};
GLboolean(EVANS_GL_CALL* LumenIsEnabled)(GLenum)=nullptr;
void(EVANS_GL_CALL* LumenGetBoolean)(GLenum,GLboolean*)=nullptr;
void(EVANS_GL_CALL* LumenGetFloat)(GLenum,GLfloat*)=nullptr;
void(EVANS_GL_CALL* LumenScissor)(GLint,GLint,GLsizei,GLsizei)=nullptr;
void(EVANS_GL_CALL* LumenBindSampler)(GLuint,GLuint)=nullptr;
void(EVANS_GL_CALL* LumenDepthRange)(GLfloat,GLfloat)=nullptr;
void(EVANS_GL_CALL* LumenFinish)()=nullptr;
void*(EVANS_GL_CALL* LumenMapBuffer)(GLenum,GLintptr,GLsizeiptr,GLbitfield)=nullptr;
GLboolean(EVANS_GL_CALL* LumenUnmapBuffer)(GLenum)=nullptr;
void(EVANS_GL_CALL* LumenGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* LumenBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;
static constexpr GLenum LumenScissorTest=0x0C11,LumenScissorBox=0x0C10,LumenDepthMask=0x0B72,LumenDepthFunction=0x0B74;
static constexpr GLenum LumenCullMode=0x0B45,LumenFrontFace=0x0B46,LumenArrayBinding=0x8894,LumenSamplerBinding=0x8919;
static constexpr GLenum LumenDepthRangeToken=0x0B70,LumenClearDepth=0x0B73,LumenStencilTest=0x0B90;
static constexpr GLenum LumenUniformStart=0x8A29,LumenUniformSize=0x8A2A,LumenStorageStart=0x90D4,LumenStorageSize=0x90D5;
static constexpr GLbitfield LumenMapRead=0x0001,LumenBufferBarrier=0x0200;
void DestroyLumenSurfaces()
{
    DestroyLumenCaptureAsync(); LumenSurfaces.clear(); for (auto& program:LumenSurfacePrograms) { if (program) GlDeleteProgram(program); program=0; } }
bool EnsureLumenSurface()
{
    GLint major=0,minor=0; GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return Fail("Lumen material capture requires OpenGL ES 3.1");
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(LumenIsEnabled,"glIsEnabled") || !LoadGL(LumenGetBoolean,"glGetBooleanv") || !LoadGL(LumenGetFloat,"glGetFloatv") ||
        !LoadGL(LumenScissor,"glScissor") || !LoadGL(LumenBindSampler,"glBindSampler") || !LoadGL(LumenDepthRange,"glDepthRangef") || !LoadGL(LumenFinish,"glFinish") ||
        !LoadGL(LumenMapBuffer,"glMapBufferRange") || !LoadGL(LumenUnmapBuffer,"glUnmapBuffer") || !LoadGL(LumenGetRange,"glGetInteger64i_v") || !LoadGL(LumenBindRange,"glBindBufferRange")) return false;
    if (LumenSurfacePrograms[0] && LumenSurfacePrograms[1]) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen material shader has no version line");
    for (unsigned stage=0;stage<LumenSurfacePrograms.size();++stage) if (!LumenSurfacePrograms[stage])
    {
        auto text=source; text.insert(version+1,stage?"#define EVANS_LUMEN_READ_STAGE 1\n":"#define EVANS_VERTEX_STAGE 1\n");
        const auto program=stage?CompileComputeProgram(text):CompileProgramPair(text,source);
        if(!program)return false;
        LumenSurfacePrograms[stage]=program;
    }
    return CheckGL("Create Lumen capture programs");
}
struct LumenCaptureState
{
    static constexpr unsigned TextureCount=5;
    Impl& Owner;
    GLint Program=0,Array=0,Vertex=0,Draw=0,Read=0,ActiveTexture=0,Unpack=0,Viewport[4]{},Scissor[4]{},DepthFunction=0,Cull=0,Front=0;
    GLint Uniform=0,Storage=0,UniformGeneric=0,StorageGeneric=0,Textures[TextureCount]{},Samplers[TextureCount]{};
    std::int64_t UniformStart=0,UniformSize=0,StorageStart=0,StorageSize=0;
    GLboolean DepthWrite=0,ColorMask[4]{}; GLfloat Range[2]{},ClearDepth=1;
    static constexpr std::array<GLenum,8> Switches{GL_DEPTH_TEST,GL_BLEND,GL_CULL_FACE,LumenScissorTest,GL_DITHER,GL_POLYGON_OFFSET_FILL,GL_SAMPLE_MASK,LumenStencilTest};
    std::array<bool,Switches.size()> Enabled{};
    explicit LumenCaptureState(Impl& owner):Owner(owner)
    {
        Owner.GlGetIntegerv(GpuCurrentProgram,&Program); Owner.GlGetIntegerv(TSRVertexArrayBinding,&Array); Owner.GlGetIntegerv(LumenArrayBinding,&Vertex);
        Owner.GlGetIntegerv(TSRDrawFramebufferBinding,&Draw); Owner.GlGetIntegerv(TSRReadFramebufferBinding,&Read); Owner.GlGetIntegerv(TSRActiveTexture,&ActiveTexture);
        Owner.GlGetIntegerv(TSRPixelUnpackBinding,&Unpack); Owner.GlGetIntegerv(TSRViewport,Viewport); Owner.GlGetIntegerv(LumenScissorBox,Scissor);
        Owner.GlGetIntegerv(LumenDepthFunction,&DepthFunction); Owner.GlGetIntegerv(LumenCullMode,&Cull); Owner.GlGetIntegerv(LumenFrontFace,&Front);
        Owner.LumenGetBoolean(LumenDepthMask,&DepthWrite); Owner.LumenGetBoolean(TSRColorMask,ColorMask);
        Owner.LumenGetFloat(LumenDepthRangeToken,Range); Owner.LumenGetFloat(LumenClearDepth,&ClearDepth);
        for (unsigned index=0;index<Switches.size();++index) Enabled[index]=Owner.LumenIsEnabled(Switches[index])!=0;
        for (unsigned slot=0;slot<TextureCount;++slot)
        { Owner.GlActiveTexture(GL_TEXTURE0+slot); Owner.GlGetIntegerv(TSRTextureBinding,&Textures[slot]); Owner.GlGetIntegerv(LumenSamplerBinding,&Samplers[slot]); }
        Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.LumenGetRange(LumenUniformStart,0,&UniformStart); Owner.LumenGetRange(LumenUniformSize,0,&UniformSize);
        Owner.GpuGetIntegerIndexed(GpuStorageBinding,0,&Storage); Owner.LumenGetRange(LumenStorageStart,0,&StorageStart); Owner.LumenGetRange(LumenStorageSize,0,&StorageSize);
        Owner.GlGetIntegerv(GpuUniformBinding,&UniformGeneric); Owner.GlGetIntegerv(GpuStorageBinding,&StorageGeneric);
    }
    ~LumenCaptureState()
    {
        Owner.GlUseProgram(Program); Owner.GlBindVertexArray(Array); Owner.GlBindBuffer(GL_ARRAY_BUFFER,Vertex);
        Owner.GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,Draw); Owner.GlBindFramebuffer(GL_READ_FRAMEBUFFER,Read); Owner.GlBindBuffer(TSRPixelUnpackBuffer,Unpack);
        Owner.GlViewport(Viewport[0],Viewport[1],Viewport[2],Viewport[3]); Owner.LumenScissor(Scissor[0],Scissor[1],Scissor[2],Scissor[3]);
        Owner.GlDepthFunc(DepthFunction); Owner.GlCullFace(Cull); Owner.GlFrontFace(Front); Owner.GlDepthMask(DepthWrite);
        Owner.GlColorMask(ColorMask[0],ColorMask[1],ColorMask[2],ColorMask[3]); Owner.LumenDepthRange(Range[0],Range[1]); Owner.GlClearDepthf(ClearDepth);
        for (unsigned index=0;index<Switches.size();++index) (Enabled[index]?Owner.GlEnable:Owner.GlDisable)(Switches[index]);
        for (unsigned slot=0;slot<TextureCount;++slot)
        { Owner.GlActiveTexture(GL_TEXTURE0+slot); Owner.GlBindTexture(GL_TEXTURE_2D,Textures[slot]); Owner.LumenBindSampler(slot,Samplers[slot]); }
        Owner.GlActiveTexture(ActiveTexture);
        if (Uniform && UniformSize>0) Owner.LumenBindRange(GL_UNIFORM_BUFFER,0,Uniform,GLintptr(UniformStart),GLsizeiptr(UniformSize));
        else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,Uniform);
        if (Storage && StorageSize>0) Owner.LumenBindRange(GL_SHADER_STORAGE_BUFFER,0,Storage,GLintptr(StorageStart),GLsizeiptr(StorageSize));
        else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,Storage);
        Owner.GlBindBuffer(GL_UNIFORM_BUFFER,UniformGeneric); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,StorageGeneric);
    }
};
struct LumenCaptureBuffers
{
    Impl& Owner; std::array<GLuint,2> Buffers{}; GLuint Array=0;
    explicit LumenCaptureBuffers(Impl& owner):Owner(owner) {}
    ~LumenCaptureBuffers() { if (Array) Owner.GlDeleteVertexArrays(1,&Array); Owner.GlDeleteBuffers(GLsizei(Buffers.size()),Buffers.data()); }
};
bool ReadLumenSurfaceRecord(const LumenSurfaceRecord& record,std::vector<LumenSurfaceTexel>& output)
{
    const auto& layout=record.Layout; const auto count=std::size_t(layout.Width)*layout.Height,bytes=count*sizeof(LumenSurfaceTexel);
    GLint limit=0,groupsX=0,groupsY=0,width=0,invocations=0;
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&limit); GlGetIntegerv(VirtualInvocations,&invocations);
    GpuGetIntegerIndexed(VirtualGroupCount,0,&groupsX); GpuGetIntegerIndexed(VirtualGroupCount,1,&groupsY); GpuGetIntegerIndexed(VirtualGroupSize,0,&width);
    const auto groups=unsigned((count+LumenSurfaceReadThreads-1)/LumenSurfaceReadThreads);
    if (limit<=0 || bytes>std::size_t(limit) || width<int(LumenSurfaceReadThreads) || invocations<int(LumenSurfaceReadThreads) ||
        groupsX<=0 || groupsY<=0 || std::min(groups,LumenSurfaceReadColumns)>unsigned(groupsX) || (groups+LumenSurfaceReadColumns-1)/LumenSurfaceReadColumns>unsigned(groupsY))
        return Fail("OpenGL ES Lumen material export exceeds compute/storage limits");
    LumenCaptureBuffers buffers(*this); GlGenBuffers(GLsizei(buffers.Buffers.size()),buffers.Buffers.data());
    const ShaderUInt4 extent{layout.Width,layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(extent),&extent,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Buffers[0]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers.Buffers[1]); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffers.Buffers[1]);
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlUseProgram(LumenSurfacePrograms[1]);
    const char* names[]{"AlbedoImage","NormalImage","EmissionImage"};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    { GlActiveTexture(GL_TEXTURE0+plane); GlBindTexture(GL_TEXTURE_2D,record.Images[plane]); LumenBindSampler(plane,0); GlUniform1i(GlGetUniformLocation(LumenSurfacePrograms[1],names[plane]),GLint(plane)); }
    if (!CheckGL("Prepare Lumen material export")) return false;
    GpuDispatch(std::min(groups,LumenSurfaceReadColumns),(groups+LumenSurfaceReadColumns-1)/LumenSurfaceReadColumns,1); GpuMemoryBarrier(GpuStorageBarrier|LumenBufferBarrier);
    if (!CheckGL("Export Lumen materials")) return false;
    std::vector<LumenSurfaceTexel> result(count); const auto* mapped=LumenMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),LumenMapRead);
    if (!mapped) return Fail("OpenGL ES Lumen material mapping failed"); std::memcpy(result.data(),mapped,bytes);
    if (!LumenUnmapBuffer(GL_SHADER_STORAGE_BUFFER) || !CheckGL("Read Lumen material export")) return Fail("OpenGL ES Lumen material export was invalidated");
    if (!ValidateLumenSurfaceTexels(layout,result,LastError)) return false; output=std::move(result); return true;
}
bool CaptureLumenSurface(const LumenSurfaceUpload& upload,LumenSurfaceResult& output)
{
    LumenSurfaceLayout layout; if (!ValidateLumenSurfaceUpload(upload,layout,LastError) || !EnsureLumenSurface()) return false;
    if (layout.Width>unsigned(MaximumTextureSize) || layout.Height>unsigned(MaximumTextureSize)) return Fail("Lumen material atlas exceeds OpenGL ES texture dimensions");
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid"); }
    }
    LumenCaptureState state(*this); LumenCaptureBuffers buffers(*this);
    auto record=std::make_shared<LumenSurfaceRecord>(*this); record->Layout=layout; record->Identity=upload.Identity; record->Captures=upload.Cards;
    GlBindBuffer(TSRPixelUnpackBuffer,0); GlActiveTexture(GL_TEXTURE0); LumenBindSampler(0,0);
    GlGenFramebuffers(1,&record->Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,record->Framebuffer);
    GlGenTextures(LumenSurfacePlanes,record->Images.data()); GLenum targets[LumenSurfacePlanes];
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        GlBindTexture(GL_TEXTURE_2D,record->Images[plane]); GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,layout.Width,layout.Height,0,GL_RGBA,GL_FLOAT,nullptr);
        GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        targets[plane]=GL_COLOR_ATTACHMENT0+plane; GlFramebufferTexture2D(GL_FRAMEBUFFER,targets[plane],GL_TEXTURE_2D,record->Images[plane],0);
    }
    GlGenTextures(1,&record->Depth); GlBindTexture(GL_TEXTURE_2D,record->Depth);
    GlTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,layout.Width,layout.Height,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,record->Depth,0); GlDrawBuffers(LumenSurfacePlanes,targets);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create Lumen material atlas")) return Fail("OpenGL ES requires three renderable FP32 Lumen material targets");
    for (auto flag:{LumenScissorTest,GL_BLEND,GL_DITHER,GL_POLYGON_OFFSET_FILL,GL_SAMPLE_MASK,LumenStencilTest}) GlDisable(flag);
    GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_LEQUAL); GlDepthMask(GL_TRUE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); LumenDepthRange(0,1);
    const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) GlClearBufferfv(GL_COLOR,GLint(plane),clear[plane]);
    GlClearDepthf(1); GlClear(GL_DEPTH_BUFFER_BIT); GlEnable(LumenScissorTest); GlCullFace(GL_BACK);
    GlGenVertexArrays(1,&buffers.Array); GlBindVertexArray(buffers.Array); GlGenBuffers(1,&buffers.Buffers[0]);
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(LumenSurfaceConstants),nullptr,GL_DYNAMIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Buffers[0]);
    GlUseProgram(LumenSurfacePrograms[0]); const char* names[]{"DiffuseImage","NormalImage","OpacityImage"};
    for (unsigned slot=0;slot<LumenSurfacePlanes;++slot) { GlUniform1i(GlGetUniformLocation(LumenSurfacePrograms[0],names[slot]),GLint(slot)); LumenBindSampler(slot,0); }
    for (GLuint attribute=0;attribute<4;++attribute) GlEnableVertexAttribArray(attribute);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2; LumenSurfaceResult result; result.Layout=layout; result.Identity=upload.Identity;
    for (unsigned card=0;card<upload.Cards.size();++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=layout.Height-(card/layout.Columns+1)*stride;
        GlViewport(x,y,stride,stride); LumenScissor(x,y,stride,stride); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; LumenSurfaceConstants constants; if (!BuildLumenSurfaceConstants(layout,capture,draw,constants,LastError)) return false;
            GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants); (draw.BackFaceCulling?GlEnable:GlDisable)(GL_CULL_FACE); GlFrontFace(RealtimeWorldIsMirrored(draw.World)?GL_CW:GL_CCW);
            GlBindBuffer(GL_ARRAY_BUFFER,Buffers.at(draw.Vertices.Value).Buffer); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,Buffers.at(draw.Indices.Value).Buffer);
            GlVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Position)));
            GlVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Normal)));
            GlVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Color)));
            GlVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,TexCoord)));
            const TextureHandle textures[]{draw.Diffuse,draw.Normal,draw.Opacity};
            for (unsigned slot=0;slot<LumenSurfacePlanes;++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,textures[slot]?Textures.at(textures[slot].Value).Texture:0); }
            GlDrawElements(GL_TRIANGLES,draw.IndexCount,GL_UNSIGNED_INT,reinterpret_cast<const void*>(std::size_t(draw.FirstIndex)*sizeof(std::uint32_t))); ++result.DrawCalls;
        }
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,0); if (!CheckGL("Capture Lumen materials")) return false;
    if (upload.Readback) { if (!ReadLumenSurfaceRecord(*record,result.Texels)) return false; }
    else { LumenFinish(); if (!CheckGL("Complete Lumen material capture")) return false; }
    result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space is exhausted");
    LumenSurfaces.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenSurface(LumenSurfaceHandle handle,std::vector<LumenSurfaceTexel>& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Lumen surface handle is invalid or expired");
    if (!EnsureLumenSurface()) return false; LumenCaptureState state(*this); return ReadLumenSurfaceRecord(*found->second,output);
}
bool DestroyLumenSurface(LumenSurfaceHandle handle)
{ return (handle && LumenSurfaces.erase(handle.Value)) || Fail("Lumen surface handle is invalid or already released"); }
