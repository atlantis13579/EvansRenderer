// ES 3.1 vertex/fragment SSBO reads share a packed immutable triangle table.
// Integer clears use glClearBufferuiv; FBO writes precede material texture reads.
static constexpr GLenum VisibilityFramebufferBinding = 0x8CA6, VisibilityTexture2DBinding = 0x8069;
static constexpr GLenum VisibilityR32UI = 0x8236, VisibilityRedInteger = 0x8D94;
static constexpr GLenum VisibilityMaxVertexBlocks = 0x90D6, VisibilityCurrentProgram = 0x8B8D;
static constexpr GLenum VisibilityStorageBinding = 0x90D3, VisibilityUniformBinding = 0x8A28;
std::array<GLuint, 2> VisibilityImages{};
std::array<GLuint, 4> VisibilityPrograms{};
GLuint VisibilityVertexArray = 0;
GLuint VisibilityFramebuffer = 0, VisibilityGeometry = 0, VisibilityUniforms = 0, VisibilityDepth = 0;
int VisibilityWidth = 0, VisibilityHeight = 0;
std::size_t VisibilityMaximumBytes = 0, VisibilityBankBytes = 0;
GLuint VisibilityOverflow = 0;
bool VisibilitySplitStorage = false;
static constexpr GLuint VisibilityOverflowBinding = 4;
// Two independently bound blocks cover the Nanite triangle budget on devices
// with 128 MiB blocks. Global triangle IDs and the shared 192-byte ABI stay intact.
std::string VisibilityShaderDefines() const
{
    return VisibilitySplitStorage ? "#define EVANS_VISIBILITY_BANK_WORDS " +
        std::to_string(VisibilityBankBytes / sizeof(unsigned)) + "u\n" : "";
}
void BindVisibilityGeometry(GLuint firstBinding)
{
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, firstBinding, VisibilityGeometry);
    if (VisibilitySplitStorage) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, VisibilityOverflowBinding, VisibilityOverflow);
}
bool AllocateVisibilityGeometry(std::size_t bytes, const void* data, GLenum usage)
{
    if (bytes > VisibilityBankBytes * (VisibilitySplitStorage ? 2 : 1))
        return Fail("ES visibility table exceeds available storage blocks");
    const auto firstBytes = std::min(bytes, VisibilityBankBytes), remaining = bytes - firstBytes;
    // Replacing either allocation invalidates the cached upload, even on failure.
    UploadedVisibility.reset();
    GLint previous = 0; GlGetIntegerv(VisibilityStorageBinding, &previous);
    if (!VisibilityGeometry) GlGenBuffers(1, &VisibilityGeometry);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER, VisibilityGeometry); GlBufferData(GL_SHADER_STORAGE_BUFFER, firstBytes, data, usage);
    if (VisibilitySplitStorage)
    {
        if (!VisibilityOverflow) GlGenBuffers(1, &VisibilityOverflow);
        // A real empty block keeps the second shader binding valid for small scenes.
        const ShaderUInt4 empty{};
        const void* tail = remaining ? (data ? static_cast<const std::uint8_t*>(data) + firstBytes : nullptr) : static_cast<const void*>(&empty);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER, VisibilityOverflow);
        GlBufferData(GL_SHADER_STORAGE_BUFFER, std::max(remaining, sizeof(empty)), tail, usage);
    }
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER, GLuint(previous));
    return CheckGL("Allocate visibility triangle blocks");
}
bool VisibilitySupported = false;
std::shared_ptr<const VisibilityScene> UploadedVisibility;
void(EVANS_GL_CALL* VisibilityClearUnsigned)(GLenum, GLint, const GLuint*) = nullptr;
void(EVANS_GL_CALL* VisibilityGetIndexed)(GLenum, GLuint, GLint*) = nullptr;
void(EVANS_GL_CALL* VisibilityGetRange)(GLenum,GLuint,std::int64_t*)=nullptr;
void(EVANS_GL_CALL* VisibilityBindRange)(GLenum,GLuint,GLuint,GLintptr,GLsizeiptr)=nullptr;

bool ProbeVisibilityBuffer()
{
    if (!MultisampleAPI || MaximumDrawBuffers < GBufferAttachmentCount || MaximumColorAttachments < GBufferAttachmentCount ||
        !HalfFloatTargetsSupported) return false;
    GLint vertexBlocks = 0, fragmentBlocks = 0, bytes = 0;
    GlGetIntegerv(VisibilityMaxVertexBlocks, &vertexBlocks); GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &fragmentBlocks);
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &bytes);
    if (vertexBlocks < 1 || fragmentBlocks < 1 || bytes < sizeof(VisibilityTriangle)) return false;
    VisibilityMaximumBytes = static_cast<std::size_t>(bytes);
    VisibilityBankBytes = VisibilityMaximumBytes / sizeof(ShaderUInt4) * sizeof(ShaderUInt4);
    GLint computeBlocks = 0, bindings = 0;
    GlGetIntegerv(TessellationComputeBlocks, &computeBlocks); GlGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &bindings);
    // Hybrid raster uses three vertex blocks; instance assembly uses four compute blocks.
    VisibilitySplitStorage = vertexBlocks >= 4 && fragmentBlocks >= 3 && computeBlocks >= 5 && bindings > GLint(VisibilityOverflowBinding);
    if (!LoadGL(VisibilityClearUnsigned, "glClearBufferuiv") || !LoadGL(VisibilityGetIndexed, "glGetIntegeri_v") || !LoadGL(VisibilityGetRange,"glGetInteger64i_v") || !LoadGL(VisibilityBindRange,"glBindBufferRange")) return false;
    GLint previousFramebuffer = 0, previousTexture = 0;
    GlGetIntegerv(VisibilityFramebufferBinding, &previousFramebuffer); GlGetIntegerv(VisibilityTexture2DBinding, &previousTexture);
    GLuint framebuffer = 0, textures[2]{}; GlGenFramebuffers(1, &framebuffer); GlGenTextures(2, textures);
    GlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    const GLenum attachments[] {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 + 1};
    for (std::size_t index = 0; index < 2; ++index)
    {
        GlBindTexture(GL_TEXTURE_2D, textures[index]); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D, 0, index ? GL_RG32F : VisibilityR32UI, 1, 1, 0,
            index ? GL_RG : VisibilityRedInteger, index ? GL_FLOAT : GL_UNSIGNED_INT, nullptr);
        GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[index], GL_TEXTURE_2D, textures[index], 0);
    }
    GlDrawBuffers(2, attachments);
    const bool complete = GlCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    bool clean = true; while (GlGetError() != GL_NO_ERROR) clean = false;
    GlBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
    GlBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previousTexture));
    GlDeleteFramebuffers(1, &framebuffer); GlDeleteTextures(2, textures);
    return complete && clean;
}
void DestroyVisibilityTargets()
{
    if (GlDeleteFramebuffers && VisibilityFramebuffer) GlDeleteFramebuffers(1, &VisibilityFramebuffer);
    if (GlDeleteTextures) GlDeleteTextures(2, VisibilityImages.data());
    VisibilityImages = {}; VisibilityFramebuffer = VisibilityDepth = 0; VisibilityWidth = VisibilityHeight = 0;
}
void DestroyVisibilityBuffer()
{
    DestroyNaniteMaterials();
    DestroyNaniteFrameRaster();
    DestroyNaniteVisibilityGeometry();
    DestroyVisibilityTargets();
    if (GlDeleteVertexArrays && VisibilityVertexArray) GlDeleteVertexArrays(1, &VisibilityVertexArray);
    VisibilityVertexArray = 0;
    if (GlDeleteBuffers) { const GLuint buffers[] {VisibilityGeometry, VisibilityOverflow, VisibilityUniforms}; GlDeleteBuffers(3, buffers); }
    if (GlDeleteProgram) for (auto program : VisibilityPrograms) if (program) GlDeleteProgram(program);
    VisibilityGeometry = VisibilityOverflow = VisibilityUniforms = 0; VisibilityPrograms = {}; UploadedVisibility.reset();
}
bool PrepareVisibilityBuffer(const std::shared_ptr<const VisibilityScene>& scene)
{
    if (!VisibilitySupported || !scene) return Fail("ES visibility requires vertex/fragment SSBOs and integer/float MRTs");
    const auto bytes = std::max(scene->TriangleCount(), std::size_t{1}) * sizeof(VisibilityTriangle);
    if (scene->HasNativeGeometry())
    {
        if (!PrepareNaniteVisibilityGeometry(*scene)) return false;
    }
    else if (UploadedVisibility != scene)
    {
        if (!ValidateVisibilityScene(*scene, LastError)) return false;
        const VisibilityTriangle empty{};
        if (!AllocateVisibilityGeometry(bytes, scene->Triangles.empty() ? &empty : scene->Triangles.data(), GL_STATIC_DRAW)) return false;
        UploadedVisibility = scene;
    }
    if (!VisibilityVertexArray) GlGenVertexArrays(1, &VisibilityVertexArray);
    if (!VisibilityUniforms) GlGenBuffers(1, &VisibilityUniforms);
    if (!VisibilityPrograms[0]) VisibilityPrograms[0] = CompileStageProgram("VisibilityBuffer.glsl", VisibilityShaderDefines().c_str());
    if (!VisibilityPrograms[1]) VisibilityPrograms[1] = CompileStageProgram("VisibilityBuffer.glsl", (VisibilityShaderDefines() + "#define EVANS_VISIBILITY_RESOLVE 1\n").c_str());
    if (!VisibilityPrograms[2]) VisibilityPrograms[2] = CompileStageProgram("VisibilityBuffer.glsl", (VisibilityShaderDefines() + "#define EVANS_VISIBILITY_WIREFRAME 1\n").c_str());
    if (!VisibilityPrograms[0] || !VisibilityPrograms[1] || !VisibilityPrograms[2]) return false;
    if (VisibilityWidth != CreateInfo.Width || VisibilityHeight != CreateInfo.Height || VisibilityDepth != DepthTexture)
    {
        DestroyVisibilityTargets(); GlGenFramebuffers(1, &VisibilityFramebuffer); GlGenTextures(2, VisibilityImages.data());
        GlBindFramebuffer(GL_FRAMEBUFFER, VisibilityFramebuffer);
        const GLenum attachments[] {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 + 1};
        for (std::size_t index = 0; index < 2; ++index)
        {
            GlBindTexture(GL_TEXTURE_2D, VisibilityImages[index]); ConfigureDepthTexture();
            GlTexImage2D(GL_TEXTURE_2D, 0, index ? GL_RG32F : VisibilityR32UI, CreateInfo.Width, CreateInfo.Height, 0,
                index ? GL_RG : VisibilityRedInteger, index ? GL_FLOAT : GL_UNSIGNED_INT, nullptr);
            GlFramebufferTexture2D(GL_FRAMEBUFFER, attachments[index], GL_TEXTURE_2D, VisibilityImages[index], 0);
        }
        GlFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthTexture, 0); GlDrawBuffers(2, attachments);
        if (GlCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return Fail("ES visibility framebuffer is incomplete");
        VisibilityWidth = CreateInfo.Width; VisibilityHeight = CreateInfo.Height; VisibilityDepth = DepthTexture;
        GlBindFramebuffer(GL_FRAMEBUFFER, 0); GlBindTexture(GL_TEXTURE_2D, 0);
    }
    return CheckGL("Prepare visibility resources");
}
bool DrawVisibilityBuffer(const VisibilityBufferDesc& frame)
{
    GLint previousProgram = 0, previousUniform = 0, previousStorage = 0, uniformSlot = 0;
    GlGetIntegerv(VisibilityCurrentProgram, &previousProgram); GlGetIntegerv(VisibilityUniformBinding, &previousUniform);
    GlGetIntegerv(VisibilityStorageBinding, &previousStorage);
    VisibilityGetIndexed(VisibilityUniformBinding, 0, &uniformSlot);
    std::array<GLint,5> savedStorage{}; std::array<std::int64_t,5> savedStart{},savedSize{};
    const GLuint storageSlots = VisibilitySplitStorage ? 5 : 3;
    for (GLuint slot=0;slot<storageSlots;++slot)
    { VisibilityGetIndexed(VisibilityStorageBinding,slot,&savedStorage[slot]); VisibilityGetRange(NaniteStorageStart,slot,&savedStart[slot]); VisibilityGetRange(NaniteStorageSize,slot,&savedSize[slot]); }
    constexpr GLenum DrawIndirectBinding=0x8F43; GLint previousIndirect=0; GlGetIntegerv(DrawIndirectBinding,&previousIndirect);
    GLint tileSampler=0;
    if (MaterialPlan.Active()) { GlActiveTexture(GL_TEXTURE0+5); GlGetIntegerv(NaniteSamplerBinding,&tileSampler); MaterialBindSampler(5,0); }
    const auto restore = [&]
    {
        if (MaterialPlan.Active()) MaterialBindSampler(5,GLuint(tileSampler));
        UnbindSampledTextures(); GlUseProgram(static_cast<GLuint>(previousProgram));
        GlBindBufferBase(GL_UNIFORM_BUFFER, 0, static_cast<GLuint>(uniformSlot));
        for (GLuint slot=0;slot<storageSlots;++slot)
            if (savedStorage[slot] && savedSize[slot]>0) VisibilityBindRange(GL_SHADER_STORAGE_BUFFER,slot,GLuint(savedStorage[slot]),GLintptr(savedStart[slot]),GLsizeiptr(savedSize[slot]));
            else GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,GLuint(savedStorage[slot]));
        GlBindBuffer(GpuDrawIndirectBuffer,GLuint(previousIndirect));
        GlBindBuffer(GL_UNIFORM_BUFFER, static_cast<GLuint>(previousUniform)); GlBindBuffer(GL_SHADER_STORAGE_BUFFER, static_cast<GLuint>(previousStorage));
        GlBindFramebuffer(GL_FRAMEBUFFER, GBufferFramebuffer); GlBindVertexArray(VertexArray); GlDepthMask(GL_TRUE_VALUE);
    };
    struct Restore { const decltype(restore)& Function; ~Restore() { Function(); } } cleanup{restore};
    UnbindSampledTextures(); // Vertex pulling owns an empty VAO without ordinary mesh attribute bindings.
    GlBindFramebuffer(GL_FRAMEBUFFER, VisibilityFramebuffer); GlBindVertexArray(VisibilityVertexArray);
    GlViewport(0, 0, CreateInfo.Width, CreateInfo.Height); GlDisable(GL_BLEND); GlDisable(GL_POLYGON_OFFSET_FILL); GlDisable(GL_DITHER);
    GlColorMask(GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE, GL_TRUE_VALUE);
    const GLuint emptyID[4]{}; const GLfloat emptyBary[4]{};
    VisibilityClearUnsigned(GL_COLOR, 0, emptyID); GlClearBufferfv(GL_COLOR, 1, emptyBary);
    BindVisibilityGeometry(0);
    const auto bind = [&](const VisibilityMaterialDraw& draw, bool resolve)
    {
        const GLuint program = VisibilityPrograms[resolve ? (MaterialPlan.Active()?3:1) : draw.Pipeline.Fill == FillMode::Wireframe ? 2 : 0]; GlUseProgram(program);
        const bool wireCull=!resolve && draw.Pipeline.Fill==FillMode::Wireframe && draw.Pipeline.BackFaceCulling;
        GlUniform1i(GlGetUniformLocation(program,"WireFaceSign"),wireCull?(frame.Scene->Draws[draw.Range].Mirrored?-1:1):0);
        GLuint textures[] {resolve ? VisibilityImages[0] : 0, resolve ? VisibilityImages[1] : 0, 0, 0, 0};
        const TextureHandle handles[] {draw.Pipeline.DiffuseTexture, draw.Pipeline.NormalTexture, draw.Pipeline.OpacityTexture};
        for (std::size_t image = 0; image < 3; ++image) if (handles[image])
        {
            const auto found = Textures.find(handles[image].Value);
            if (found == Textures.end() || found->second.Description.Format != TextureFormat::RGBA8) return Fail("ES visibility material texture is invalid");
            textures[image + 2] = found->second.Texture;
        }
        VisibilityConstants constants;
        if (!BuildVisibilityConstants(frame.ViewProjection, draw.Pipeline, CreateInfo.Width, CreateInfo.Height,
            frame.Scene->TriangleCount(), frame.Scene->Draws[draw.Range], textures[2] != 0, textures[3] != 0, textures[4] != 0, constants))
            return Fail("ES visibility constants are invalid");
        constants.MaterialTiles=MaterialPlan.Resolve(unsigned(&draw-frame.Draws.data()));
        if (MaterialPlan.Active()) constants.MaterialTiles.W=MaterialPlan.Constants.Layout.X;
        GlBindBuffer(GL_UNIFORM_BUFFER, VisibilityUniforms); GlBufferData(GL_UNIFORM_BUFFER, sizeof(constants), &constants, GL_DYNAMIC_DRAW);
        GlBindBufferBase(GL_UNIFORM_BUFFER, 0, VisibilityUniforms);
        const char* names[] {"VisibilityImage", "BarycentricImage", "AlbedoImage", "NormalImage", "OpacityImage"};
        for (GLuint image = 0; image < 5; ++image)
        { GlActiveTexture(GL_TEXTURE0 + image); GlBindTexture(GL_TEXTURE_2D, textures[image]); GlUniform1i(GlGetUniformLocation(program, names[image]), image); }
        if (resolve && MaterialPlan.Active())
        { GlActiveTexture(GL_TEXTURE0+5); GlBindTexture(GL_TEXTURE_2D,MaterialTileImage); GlUniform1i(GlGetUniformLocation(program,"MaterialScheduleImage"),5); }
        return true;
    };
    if (frame.NaniteRaster) { if (!DrawNaniteFrameRaster(frame)) return false; }
    else
    for (const auto& draw : frame.Draws)
    {
        if (!bind(draw, false)) return false;
        const auto& pipeline = draw.Pipeline; const auto& range = frame.Scene->Draws[draw.Range];
        if (pipeline.DepthTest) GlEnable(GL_DEPTH_TEST); else GlDisable(GL_DEPTH_TEST);
        GlDepthFunc(pipeline.DepthWrite ? GL_LESS : GL_LEQUAL); GlDepthMask(pipeline.DepthWrite ? GL_TRUE_VALUE : GL_FALSE_VALUE);
        const bool wire = pipeline.Fill == FillMode::Wireframe;
        if (pipeline.BackFaceCulling && !wire) GlEnable(GL_CULL_FACE); else GlDisable(GL_CULL_FACE);
        GlCullFace(GL_BACK); GlFrontFace(range.Mirrored ? GL_CW : GL_CCW);
        const auto vertices = std::uint64_t(range.TriangleCount) * (wire ? 6u : VisibilityTriangleCorners);
        if (vertices > std::numeric_limits<GLsizei>::max()) return Fail("ES visibility draw exceeds signed vertex addressing");
        GlDrawArrays(wire ? GL_LINES : GL_TRIANGLES, 0, static_cast<GLsizei>(vertices));
    }
    if (!ClassifyNaniteMaterials()) return false;
    UnbindSampledTextures(); GlBindFramebuffer(GL_FRAMEBUFFER, GBufferFramebuffer);
    BindVisibilityGeometry(0);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,MaterialMap); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,MaterialWork);
    GlDisable(GL_DEPTH_TEST); GlDepthMask(GL_FALSE_VALUE); GlDisable(GL_CULL_FACE);
    if (MaterialPlan.Active())
    {
        GlBindBuffer(GpuDrawIndirectBuffer,MaterialWork);
        for (unsigned bin=0;bin<MaterialPlan.Representatives.size();++bin)
        {
            if (!bind(frame.Draws[MaterialPlan.Representatives[bin]],true)) return false;
            MaterialDrawIndirect(GL_TRIANGLES,reinterpret_cast<const void*>(std::size_t(bin)*NaniteMaterialBinWords*sizeof(unsigned)));
        }
    }
    else for (const auto& draw : frame.Draws) { if (!bind(draw, true)) return false; GlDrawArrays(GL_TRIANGLES, 0, 3); }
    return CheckGL("Rasterize visibility IDs and reconstruct material attributes");
}
