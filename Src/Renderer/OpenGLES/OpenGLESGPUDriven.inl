// ES 3.1 SSBO output is reused as an instance UBO. Memory barriers cover both
// graphics consumers; frame-local slots prevent aliasing between recorded draws.
struct GpuBatch
{
    GpuDrivenBatchDesc Description;
    GLuint Constants = 0, Input = 0, Output = 0, Arguments = 0;
};
std::vector<GpuBatch> GpuBatches;
GpuBatch* ActiveGpuBatch = nullptr;
std::size_t GpuCursor = 0;
std::uint64_t GpuEpoch = 0;
GLuint GpuCullProgram = 0;
bool GpuDrivenSupported = false;
void(EVANS_GL_CALL* GpuDispatch)(GLuint, GLuint, GLuint) = nullptr;
void(EVANS_GL_CALL* GpuMemoryBarrier)(GLbitfield) = nullptr;
void(EVANS_GL_CALL* GpuDrawElementsIndirect)(GLenum, GLenum, const void*) = nullptr;
void(EVANS_GL_CALL* GpuGetIntegerIndexed)(GLenum, GLuint, GLint*) = nullptr;
static constexpr GLenum GpuComputeShader = 0x91B9, GpuDrawIndirectBuffer = 0x8F3F;
static constexpr GLenum GpuMajorVersion = 0x821B, GpuMinorVersion = 0x821C, GpuCurrentProgram = 0x8B8D;
static constexpr GLenum GpuStorageBinding = 0x90D3, GpuUniformBinding = 0x8A28;
static constexpr GLbitfield GpuUniformBarrier = 0x00000004, GpuCommandBarrier = 0x00000040, GpuStorageBarrier = 0x00002000;

bool ProbeGpuDriven()
{
    GLint major = 0, minor = 0;
    GlGetIntegerv(GpuMajorVersion, &major); GlGetIntegerv(GpuMinorVersion, &minor);
    return (major > 3 || (major == 3 && minor >= 1)) && MaximumUniformBlockSize >= sizeof(InstanceConstantBlock) &&
        LoadGL(GpuDispatch, "glDispatchCompute") && LoadGL(GpuMemoryBarrier, "glMemoryBarrier") &&
        LoadGL(GpuDrawElementsIndirect, "glDrawElementsIndirect") && LoadGL(GpuGetIntegerIndexed, "glGetIntegeri_v");
}
void DestroyGpuDriven()
{
    if (GlDeleteBuffers) for (auto& batch : GpuBatches)
    {
        const GLuint buffers[] {batch.Constants, batch.Input, batch.Output, batch.Arguments};
        GlDeleteBuffers(4, buffers);
    }
    if (GpuCullProgram && GlDeleteProgram) GlDeleteProgram(GpuCullProgram);
    GpuCullProgram = 0; GpuCursor = 0; ActiveGpuBatch = nullptr; GpuBatches.clear();
}
bool EnsureGpuCull()
{
    if (!GpuDrivenSupported) return Fail("GPU-driven rendering requires OpenGL ES 3.1 compute and indirect drawing");
    if (GpuCullProgram) return true;
    std::string source;
    if (!LoadTechniqueShaderSource("OpenGLES", "GPUDrivenRendering.glsl", source, LastError)) return false;
    const auto program=CompileComputeProgram(source); if(!program)return false;
    GpuCullProgram = program; return true;
}
bool PrepareGpuBatch(const GpuDrivenBatchDesc& desc, std::uint32_t& index)
{
    GpuDrivenBatchUpload upload;
    if (!BuildGpuDrivenBatch(desc, upload)) return Fail("ES GPU batch contains invalid bounds, transforms or command data");
    // ES wireframe uses an expanded edge buffer; nonzero base vertices use a
    // rebased scratch index buffer. Visibility still operates on original bounds.
    if (desc.Wireframe || desc.BaseVertex != 0)
    {
        if (desc.Wireframe && desc.IndexCount > std::numeric_limits<std::uint32_t>::max() / 2)
            return Fail("ES indirect wireframe index count overflows");
        upload.Constants.Draw.X = desc.Wireframe ? desc.IndexCount * 2 : desc.IndexCount;
        upload.Constants.Draw.Y = upload.Constants.Draw.Z = 0;
    }
    if (!EnsureGpuCull()) return false;
    if (GpuCursor == GpuBatches.size())
    {
        GLuint buffers[4] {}; GlGenBuffers(4, buffers);
        GpuBatch batch; batch.Constants = buffers[0]; batch.Input = buffers[1]; batch.Output = buffers[2]; batch.Arguments = buffers[3];
        GpuBatches.push_back(std::move(batch));
    }
    auto& batch = GpuBatches[GpuCursor]; batch.Description = desc;
    GLint previousProgram = 0, previousUniform = 0, previousStorage = 0, uniformSlot = 0;
    GLint storageSlots[3] {};
    GlGetIntegerv(GpuCurrentProgram, &previousProgram); GlGetIntegerv(GpuUniformBinding, &previousUniform);
    GlGetIntegerv(GpuStorageBinding, &previousStorage); GpuGetIntegerIndexed(GpuUniformBinding, 0, &uniformSlot);
    for (GLuint slot = 0; slot < 3; ++slot) GpuGetIntegerIndexed(GpuStorageBinding, slot, &storageSlots[slot]);
    GlBindBuffer(GL_UNIFORM_BUFFER, batch.Constants); GlBufferData(GL_UNIFORM_BUFFER, sizeof(upload.Constants), &upload.Constants, GL_DYNAMIC_DRAW);
    GlBindBufferBase(GL_UNIFORM_BUFFER, 0, batch.Constants);
    const GLuint storage[] {batch.Input, batch.Output, batch.Arguments};
    const std::size_t sizes[] {sizeof(upload.Candidates), sizeof(InstanceConstantBlock), sizeof(GpuDrivenIndexedArguments)};
    for (GLuint slot = 0; slot < 3; ++slot)
    {
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER, storage[slot]);
        GlBufferData(GL_SHADER_STORAGE_BUFFER, sizes[slot], slot == 0 ? upload.Candidates.data() : nullptr, GL_DYNAMIC_DRAW);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, storage[slot]);
    }
    GlUseProgram(GpuCullProgram); GpuDispatch(1, 1, 1);
    GpuMemoryBarrier(GpuUniformBarrier | GpuCommandBarrier | GpuStorageBarrier);
    GlUseProgram(static_cast<GLuint>(previousProgram));
    for (GLuint slot = 0; slot < 3; ++slot) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, static_cast<GLuint>(storageSlots[slot]));
    GlBindBufferBase(GL_UNIFORM_BUFFER, 0, static_cast<GLuint>(uniformSlot));
    GlBindBuffer(GL_UNIFORM_BUFFER, static_cast<GLuint>(previousUniform));
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER, static_cast<GLuint>(previousStorage));
    if (!CheckGL("Compute GPU visibility and indirect arguments")) return false;
    index = static_cast<std::uint32_t>(GpuCursor++); return true;
}
