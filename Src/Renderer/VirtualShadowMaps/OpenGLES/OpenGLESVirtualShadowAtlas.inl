// Canonical page origins are top-left. Only viewport placement and native depth
// reads flip GL rows; material coverage remains in logical virtual texel space.
struct VirtualShadowAtlasRecord
{
    Impl& Owner; VirtualShadowSnapshot Info; GLuint Depth=0,Framebuffer=0,Directory=0;
    explicit VirtualShadowAtlasRecord(Impl& owner):Owner(owner) {}
    ~VirtualShadowAtlasRecord() { if (Framebuffer) Owner.GlDeleteFramebuffers(1,&Framebuffer); if (Depth) Owner.GlDeleteTextures(1,&Depth); if (Directory) Owner.GlDeleteTextures(1,&Directory); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<VirtualShadowAtlasRecord>> VirtualShadowAtlases;
std::array<GLuint,4> VirtualAtlasPrograms{};
GLboolean(EVANS_GL_CALL* VirtualAtlasIsEnabled)(GLenum)=nullptr;
void(EVANS_GL_CALL* VirtualAtlasGetBoolean)(GLenum,GLboolean*)=nullptr;
void(EVANS_GL_CALL* VirtualAtlasGetFloat)(GLenum,GLfloat*)=nullptr;
void(EVANS_GL_CALL* VirtualAtlasScissor)(GLint,GLint,GLsizei,GLsizei)=nullptr;
void(EVANS_GL_CALL* VirtualAtlasBindSampler)(GLuint,GLuint)=nullptr;
void(EVANS_GL_CALL* VirtualAtlasDepthRange)(GLfloat,GLfloat)=nullptr;
void(EVANS_GL_CALL* VirtualAtlasFinish)()=nullptr;
static constexpr GLenum AtlasScissorBox=0x0C10,AtlasScissorTest=0x0C11,AtlasDepthWriteMask=0x0B72,AtlasDepthFunction=0x0B74;
static constexpr GLenum AtlasCullMode=0x0B45,AtlasFrontFace=0x0B46,AtlasArrayBinding=0x8894,AtlasSamplerBinding=0x8919;
static constexpr GLenum AtlasDepthRange=0x0B70,AtlasClearDepth=0x0B73,AtlasCompareMode=0x884C,AtlasMaximumTexture=0x0D33;
static constexpr GLenum AtlasUnpackAlignment=0x0CF5,AtlasUnpackRowLength=0x0CF2;
void DestroyVirtualShadowAtlases()
{
    FrameVirtualShadowAtlas.reset(); FrameVirtualShadowSampling={};
    VirtualShadowAtlases.clear();
    for (auto& program:VirtualAtlasPrograms) { if (program) GlDeleteProgram(program); program=0; }
}
bool EnsureVirtualShadowAtlas()
{
    GLint major=0,minor=0; GlGetIntegerv(GpuMajorVersion,&major); GlGetIntegerv(GpuMinorVersion,&minor);
    if (major<3 || (major==3 && minor<1)) return Fail("Virtual depth atlas requires ES 3.1");
    if (!LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier") || !LoadGL(GpuGetIntegerIndexed,"glGetIntegeri_v") ||
        !LoadGL(VirtualMapBuffer,"glMapBufferRange") || !LoadGL(VirtualUnmapBuffer,"glUnmapBuffer") || !LoadGL(VirtualGetRange,"glGetInteger64i_v") || !LoadGL(VirtualBindRange,"glBindBufferRange") ||
        !LoadGL(VirtualAtlasIsEnabled,"glIsEnabled") || !LoadGL(VirtualAtlasGetBoolean,"glGetBooleanv") || !LoadGL(VirtualAtlasGetFloat,"glGetFloatv") ||
        !LoadGL(VirtualAtlasScissor,"glScissor") || !LoadGL(VirtualAtlasBindSampler,"glBindSampler") || !LoadGL(VirtualAtlasDepthRange,"glDepthRangef") || !LoadGL(VirtualAtlasFinish,"glFinish")) return false;
    if (VirtualAtlasPrograms[0] && VirtualAtlasPrograms[1] && VirtualAtlasPrograms[2] && VirtualAtlasPrograms[3]) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","VirtualShadowMaps.glsl",source,LastError)) return false;
    const auto firstLine=source.find('\n'); if (firstLine==std::string::npos) return Fail("Virtual atlas shader has no version line");
    constexpr const char* stages[]{"EVANS_VIRTUAL_SHADOW_RASTER_STAGE","EVANS_VIRTUAL_SHADOW_COPY_STAGE","EVANS_VIRTUAL_SHADOW_READ_STAGE","EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE"};
    for (unsigned stage=0;stage<VirtualAtlasPrograms.size();++stage) if (!VirtualAtlasPrograms[stage])
    {
        auto text=source; text.insert(firstLine+1,std::string("#define ")+stages[stage]+" 1\n");
        auto vertex=text; if(stage<2)vertex.insert(firstLine+1,"#define EVANS_VERTEX_STAGE 1\n");
        const auto program=stage>=2?CompileComputeProgram(text):CompileProgramPair(vertex,text);
        if(!program)return false;
        VirtualAtlasPrograms[stage]=program;
    }
    return true;
}
struct VirtualAtlasState
{
    Impl& Owner;
    GLint Program=0,Array=0,VertexBuffer=0,Draw=0,Read=0,ActiveTexture=0,Unpack=0,Viewport[4]{},Scissor[4]{},DepthFunction=0,CullMode=0,Front=0;
    GLint Uniform=0,Storage=0,StorageZero=0,UniformGeneric=0,StorageGeneric=0,Textures[3]{},Samplers[3]{},UnpackAlignment=0,UnpackRowLength=0;
    std::int64_t UniformStart=0,UniformSize=0,StorageStart=0,StorageSize=0,StorageZeroStart=0,StorageZeroSize=0;
    GLboolean DepthMask=0,ColorMask[4]{}; GLfloat Range[2]{},ClearDepth=1;
    static constexpr std::array<GLenum,7> Switches{GL_DEPTH_TEST,GL_BLEND,GL_CULL_FACE,AtlasScissorTest,GL_DITHER,GL_POLYGON_OFFSET_FILL,GL_SAMPLE_MASK};
    std::array<bool,Switches.size()> Enabled{};
    explicit VirtualAtlasState(Impl& owner):Owner(owner)
    {
        Owner.GlGetIntegerv(GpuCurrentProgram,&Program); Owner.GlGetIntegerv(TSRVertexArrayBinding,&Array); Owner.GlGetIntegerv(AtlasArrayBinding,&VertexBuffer);
        Owner.GlGetIntegerv(TSRDrawFramebufferBinding,&Draw); Owner.GlGetIntegerv(TSRReadFramebufferBinding,&Read); Owner.GlGetIntegerv(TSRActiveTexture,&ActiveTexture);
        Owner.GlGetIntegerv(TSRPixelUnpackBinding,&Unpack); Owner.GlGetIntegerv(TSRViewport,Viewport); Owner.GlGetIntegerv(AtlasScissorBox,Scissor);
        Owner.GlGetIntegerv(AtlasUnpackAlignment,&UnpackAlignment); Owner.GlGetIntegerv(AtlasUnpackRowLength,&UnpackRowLength);
        Owner.GlGetIntegerv(AtlasDepthFunction,&DepthFunction); Owner.GlGetIntegerv(AtlasCullMode,&CullMode); Owner.GlGetIntegerv(AtlasFrontFace,&Front);
        Owner.VirtualAtlasGetBoolean(AtlasDepthWriteMask,&DepthMask); Owner.VirtualAtlasGetBoolean(TSRColorMask,ColorMask);
        Owner.VirtualAtlasGetFloat(AtlasDepthRange,Range); Owner.VirtualAtlasGetFloat(AtlasClearDepth,&ClearDepth);
        for (unsigned index=0;index<Switches.size();++index) Enabled[index]=Owner.VirtualAtlasIsEnabled(Switches[index])!=0;
        for (unsigned unit=0;unit<3;++unit)
        { Owner.GlActiveTexture(GL_TEXTURE0+unit); Owner.GlGetIntegerv(TSRTextureBinding,&Textures[unit]); Owner.GlGetIntegerv(AtlasSamplerBinding,&Samplers[unit]); }
        Owner.GpuGetIntegerIndexed(GpuUniformBinding,0,&Uniform); Owner.VirtualGetRange(VirtualUniformStart,0,&UniformStart); Owner.VirtualGetRange(VirtualUniformSize,0,&UniformSize);
        Owner.GpuGetIntegerIndexed(GpuStorageBinding,1,&Storage); Owner.VirtualGetRange(VirtualStorageStart,1,&StorageStart); Owner.VirtualGetRange(VirtualStorageSize,1,&StorageSize);
        Owner.GpuGetIntegerIndexed(GpuStorageBinding,0,&StorageZero); Owner.VirtualGetRange(VirtualStorageStart,0,&StorageZeroStart); Owner.VirtualGetRange(VirtualStorageSize,0,&StorageZeroSize);
        Owner.GlGetIntegerv(GpuUniformBinding,&UniformGeneric); Owner.GlGetIntegerv(GpuStorageBinding,&StorageGeneric);
    }
    ~VirtualAtlasState()
    {
        Owner.GlUseProgram(Program); Owner.GlBindVertexArray(Array); Owner.GlBindBuffer(GL_ARRAY_BUFFER,VertexBuffer);
        Owner.GlBindFramebuffer(GL_DRAW_FRAMEBUFFER,Draw); Owner.GlBindFramebuffer(GL_READ_FRAMEBUFFER,Read); Owner.GlBindBuffer(TSRPixelUnpackBuffer,Unpack);
        Owner.GlPixelStorei(AtlasUnpackAlignment,UnpackAlignment); Owner.GlPixelStorei(AtlasUnpackRowLength,UnpackRowLength);
        Owner.GlViewport(Viewport[0],Viewport[1],Viewport[2],Viewport[3]); Owner.VirtualAtlasScissor(Scissor[0],Scissor[1],Scissor[2],Scissor[3]);
        Owner.GlDepthFunc(DepthFunction); Owner.GlCullFace(CullMode); Owner.GlFrontFace(Front); Owner.GlDepthMask(DepthMask);
        Owner.GlColorMask(ColorMask[0],ColorMask[1],ColorMask[2],ColorMask[3]); Owner.VirtualAtlasDepthRange(Range[0],Range[1]); Owner.GlClearDepthf(ClearDepth);
        for (unsigned index=0;index<Switches.size();++index) (Enabled[index]?Owner.GlEnable:Owner.GlDisable)(Switches[index]);
        for (unsigned unit=0;unit<3;++unit)
        { Owner.GlActiveTexture(GL_TEXTURE0+unit); Owner.GlBindTexture(GL_TEXTURE_2D,Textures[unit]); Owner.VirtualAtlasBindSampler(unit,Samplers[unit]); }
        Owner.GlActiveTexture(ActiveTexture);
        if (Uniform && UniformSize>0) Owner.VirtualBindRange(GL_UNIFORM_BUFFER,0,Uniform,static_cast<GLintptr>(UniformStart),static_cast<GLsizeiptr>(UniformSize));
        else Owner.GlBindBufferBase(GL_UNIFORM_BUFFER,0,Uniform);
        if (Storage && StorageSize>0) Owner.VirtualBindRange(GL_SHADER_STORAGE_BUFFER,1,Storage,static_cast<GLintptr>(StorageStart),static_cast<GLsizeiptr>(StorageSize));
        else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,Storage);
        if (StorageZero && StorageZeroSize>0) Owner.VirtualBindRange(GL_SHADER_STORAGE_BUFFER,0,StorageZero,static_cast<GLintptr>(StorageZeroStart),static_cast<GLsizeiptr>(StorageZeroSize));
        else Owner.GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,StorageZero);
        Owner.GlBindBuffer(GL_UNIFORM_BUFFER,UniformGeneric); Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,StorageGeneric);
    }
};
struct VirtualAtlasBuffers
{
    Impl& Owner; std::array<GLuint,2> Buffers{}; GLuint Array=0;
    explicit VirtualAtlasBuffers(Impl& owner):Owner(owner) {}
    ~VirtualAtlasBuffers() { if (Array) Owner.GlDeleteVertexArrays(1,&Array); Owner.GlDeleteBuffers(static_cast<GLsizei>(Buffers.size()),Buffers.data()); }
};
bool PackVirtualShadowDepth(GLuint depth,unsigned imageWidth,unsigned imageHeight,VirtualAtlasBuffers& buffers)
{
    const auto count=std::size_t(imageWidth)*imageHeight,bytes=count*sizeof(float);
    GLint limit=0,groupsX=0,groupsY=0,width=0,invocations=0;
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&limit); GlGetIntegerv(VirtualInvocations,&invocations);
    GpuGetIntegerIndexed(VirtualGroupCount,0,&groupsX); GpuGetIntegerIndexed(VirtualGroupCount,1,&groupsY); GpuGetIntegerIndexed(VirtualGroupSize,0,&width);
    const auto groups=static_cast<unsigned>((count+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth);
    if (limit<=0 || bytes>std::size_t(limit) || width<int(VirtualShadowGroupWidth) || invocations<int(VirtualShadowGroupWidth) ||
        std::min(groups,VirtualShadowGroupColumns)>unsigned(groupsX) || (groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns>unsigned(groupsY)) return Fail("ES virtual atlas feedback exceeds native limits");
    GlGenBuffers(static_cast<GLsizei>(buffers.Buffers.size()),buffers.Buffers.data());
    VirtualShadowRasterConstants constants; constants.Extent={imageWidth,imageHeight,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0};
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(constants),&constants,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Buffers[0]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers.Buffers[1]); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(bytes),nullptr,GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,buffers.Buffers[1]);
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlActiveTexture(GL_TEXTURE0+2); GlBindTexture(GL_TEXTURE_2D,depth); VirtualAtlasBindSampler(2,0);
    GlUseProgram(VirtualAtlasPrograms[2]); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[2],"ParentDepth"),2);
    if (!CheckGL("Prepare virtual atlas feedback")) return false;
    GpuDispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1); GpuMemoryBarrier(GpuStorageBarrier|VirtualBufferBarrier);
    return CheckGL("Pack virtual receiver depth");
}
bool ReadVirtualShadowAtlasRecord(const VirtualShadowAtlasRecord& record,std::vector<float>& output)
{
    const auto& layout=record.Info.Layout; const auto count=std::size_t(layout.AtlasWidth)*layout.AtlasHeight,bytes=count*sizeof(float);
    VirtualAtlasBuffers buffers(*this);
    if (!PackVirtualShadowDepth(record.Depth,layout.AtlasWidth,layout.AtlasHeight,buffers)) return false;
    const auto* mapped=VirtualMapBuffer(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(bytes),VirtualMapRead);
    if (!mapped) return Fail("ES virtual atlas feedback mapping failed");
    std::vector<float> result(count); std::memcpy(result.data(),mapped,bytes);
    if (!VirtualUnmapBuffer(GL_SHADER_STORAGE_BUFFER) || !CheckGL("Read virtual atlas depth")) return Fail("ES virtual atlas feedback was invalidated");
    if (!ValidateVirtualShadowAtlasDepth(record.Info,result,LastError)) return false; output=std::move(result); return true;
}
bool ReadVirtualShadowAtlas(VirtualShadowAtlasHandle handle,std::vector<float>& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("ES virtual atlas handle is invalid");
    if (!EnsureVirtualShadowAtlas()) return false; VirtualAtlasState state(*this); return ReadVirtualShadowAtlasRecord(*found->second,output);
}
bool DestroyVirtualShadowAtlas(VirtualShadowAtlasHandle handle)
{ return (handle && VirtualShadowAtlases.erase(handle.Value)) || Fail("ES virtual atlas handle is invalid or already released"); }
bool RenderVirtualShadowAtlas(const VirtualShadowAtlasUpload& upload,VirtualShadowAtlasHandle previous,VirtualShadowAtlasResult& output)
{
    std::shared_ptr<VirtualShadowAtlasRecord> parent;
    if (previous) { const auto found=VirtualShadowAtlases.find(previous.Value); if (found==VirtualShadowAtlases.end()) return Fail("ES virtual atlas parent is invalid"); parent=found->second; }
    VirtualShadowPlan plan;
    if (!BuildVirtualShadowPlan(upload.Identity,upload.Settings,upload.Requests,upload.Invalidations,parent?&parent->Info:nullptr,plan,LastError) || !EnsureVirtualShadowAtlas()) return false;
    for (const auto& caster:upload.Casters)
    {
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Virtual atlas caster buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        { const auto image=Textures.find(texture.Value); if (image==Textures.end() || image->second.Description.Format!=TextureFormat::RGBA8) return Fail("Virtual atlas coverage requires existing color textures"); }
    }
    const auto& layout=plan.Candidate.Layout; GLint limit=0; GlGetIntegerv(AtlasMaximumTexture,&limit);
    if (limit<=0 || layout.AtlasWidth>unsigned(limit) || layout.AtlasHeight>unsigned(limit)) return Fail("ES virtual atlas exceeds native texture dimensions");
    VirtualAtlasState state(*this); VirtualAtlasBuffers buffers(*this);
    auto record=std::make_shared<VirtualShadowAtlasRecord>(*this); record->Info=plan.Candidate;
    GlBindBuffer(TSRPixelUnpackBuffer,0); GlActiveTexture(GL_TEXTURE0+2); VirtualAtlasBindSampler(2,0);
    GlGenTextures(1,&record->Depth); GlBindTexture(GL_TEXTURE_2D,record->Depth);
    GlTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,layout.AtlasWidth,layout.AtlasHeight,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,AtlasCompareMode,GL_NONE_VALUE);
    GlGenFramebuffers(1,&record->Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,record->Framebuffer); GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,record->Depth,0);
    const GLenum none=GL_NONE_VALUE; GlDrawBuffers(1,&none); GlReadBuffer(GL_NONE_VALUE);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create immutable virtual depth atlas")) return Fail("ES virtual depth atlas is incomplete");
    GlDisable(AtlasScissorTest); GlDisable(GL_BLEND); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDisable(GL_SAMPLE_MASK);
    GlEnable(GL_DEPTH_TEST); GlDepthMask(GL_TRUE_VALUE); GlColorMask(GL_FALSE_VALUE,GL_FALSE_VALUE,GL_FALSE_VALUE,GL_FALSE_VALUE); VirtualAtlasDepthRange(0,1);
    GlClearDepthf(1); GlClear(GL_DEPTH_BUFFER_BIT); GlEnable(AtlasScissorTest);
    GlGenVertexArrays(1,&buffers.Array); GlBindVertexArray(buffers.Array); GlGenBuffers(1,&buffers.Buffers[0]);
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(VirtualShadowRasterConstants),nullptr,GL_DYNAMIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Buffers[0]);
    const auto stride=layout.PageSide+VirtualShadowPageBorder*2;
    const auto frame=[&](const VirtualShadowRasterConstants& constants)
    {
        const auto x=static_cast<GLint>(constants.Origins.X),y=static_cast<GLint>(layout.AtlasHeight-stride)-static_cast<GLint>(constants.Origins.Y);
        GlViewport(x,y,stride,stride); VirtualAtlasScissor(x,y,stride,stride); GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants);
    };
    GlUseProgram(VirtualAtlasPrograms[1]); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[1],"ParentDepth"),2);
    GlActiveTexture(GL_TEXTURE0+2); GlBindTexture(GL_TEXTURE_2D,parent?parent->Depth:0); GlDisable(GL_CULL_FACE); GlDepthFunc(GL_ALWAYS);
    for (const auto& copy:plan.Copies) { frame(BuildVirtualShadowCopyConstants(parent->Info.Layout,layout,copy)); GlDrawArrays(GL_TRIANGLES,0,3); }
    GlBindTexture(GL_TEXTURE_2D,0); GlUseProgram(VirtualAtlasPrograms[0]); GlDepthFunc(GL_LEQUAL); GlCullFace(GL_BACK);
    GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[0],"DiffuseImage"),0); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[0],"OpacityImage"),1);
    for (GLuint attribute:{0U,2U,3U}) GlEnableVertexAttribArray(attribute);
    VirtualAtlasBindSampler(0,0); VirtualAtlasBindSampler(1,0);
    VirtualShadowAtlasResult result;
    for (const auto& tile:plan.Dirty) for (const auto& caster:upload.Casters)
    {
        VirtualShadowRasterConstants constants; if (!BuildVirtualShadowRasterConstants(record->Info,tile,caster,constants,LastError)) return false; frame(constants);
        (caster.BackFaceCulling?GlEnable:GlDisable)(GL_CULL_FACE); GlFrontFace(RealtimeWorldIsMirrored(caster.World)?GL_CW:GL_CCW);
        GlBindBuffer(GL_ARRAY_BUFFER,Buffers.at(caster.Vertices.Value).Buffer); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,Buffers.at(caster.Indices.Value).Buffer);
        GlVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Position)));
        GlVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Color)));
        GlVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,TexCoord)));
        GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,caster.Diffuse?Textures.at(caster.Diffuse.Value).Texture:0);
        GlActiveTexture(GL_TEXTURE0+1); GlBindTexture(GL_TEXTURE_2D,caster.Opacity?Textures.at(caster.Opacity.Value).Texture:0);
        GlDrawElements(GL_TRIANGLES,caster.IndexCount,GL_UNSIGNED_INT,reinterpret_cast<const void*>(std::size_t(caster.FirstIndex)*sizeof(std::uint32_t))); ++result.DrawCalls;
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,0); if (!CheckGL("Draw virtual shadow pages")) return false;
    if (upload.Readback) { if (!ReadVirtualShadowAtlasRecord(*record,result.Depth)) return false; }
    else { VirtualAtlasFinish(); if (!CheckGL("Complete virtual depth atlas")) return false; }
    std::vector<unsigned> rendered; for (const auto& tile:plan.Dirty) rendered.push_back(tile.Page);
    if (!CompleteVirtualShadowPlan(plan,rendered,record->Info,LastError)) return false;
    if (!CreateVirtualShadowDirectory(*record)) return false;
    result.Handle=NewVirtualShadowAtlasHandle(); if (!result.Handle) return Fail("Virtual atlas identity space is exhausted");
    result.Info=record->Info; result.CopiedPages=static_cast<unsigned>(plan.Copies.size()); result.RenderedPages=static_cast<unsigned>(plan.Dirty.size());
    VirtualShadowAtlases.emplace(result.Handle.Value,record); output=std::move(result); return true;
}
