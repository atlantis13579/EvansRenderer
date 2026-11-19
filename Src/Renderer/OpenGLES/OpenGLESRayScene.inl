// Fragment SSBO support is optional in ES 3.1. Check the actual stage limits,
// then publish immutable scene storage only after all four uploads succeed.
std::shared_ptr<const RaySceneUpload> FrameRayScene, UploadedRayScene;
std::array<GLuint,RaySceneBufferCount> RaySceneBuffers{};
GLuint RayAOProgram=0;
bool RaySceneSupported=false;

void DestroyRayScene()
{
    DestroyRayRadiance();
    if (GlDeleteBuffers) GlDeleteBuffers(static_cast<GLsizei>(RaySceneBuffers.size()),RaySceneBuffers.data());
    RaySceneBuffers={}; FrameRayScene.reset(); UploadedRayScene.reset();
    if (RayAOProgram && GlDeleteProgram) GlDeleteProgram(RayAOProgram);
    RayAOProgram=0; RaySceneSupported=false;
}

bool EnsureRayScene()
{
    if (!FrameRayScene || !RaySceneSupported) return Fail("OpenGL ES ray AO requires fragment storage and a scene snapshot");
    if (FrameRayScene==UploadedRayScene) return true;
    if (!ValidateRaySceneUpload(*FrameRayScene,LastError)) return false;
    GLint blocks=0,bytes=0;
    GlGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&blocks);
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&bytes);
    if (!CheckGL("Query ray-scene storage limits") || blocks<static_cast<GLint>(RaySceneBufferCount) || bytes<=0)
        return Fail("OpenGL ES ray AO requires four fragment storage blocks");
    const auto data=GetRaySceneBufferData(*FrameRayScene);
    for (const auto& buffer:data) if (buffer.Bytes>static_cast<std::size_t>(bytes))
        return Fail("OpenGL ES ray scene exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE");
    std::array<GLuint,RaySceneBufferCount> buffers{};
    GlGenBuffers(static_cast<GLsizei>(buffers.size()),buffers.data());
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[slot]);
        GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(data[slot].Bytes),data[slot].Data,GL_STATIC_DRAW);
    }
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,0);
    if (!CheckGL("Upload ray scene"))
    {
        GlDeleteBuffers(static_cast<GLsizei>(buffers.size()),buffers.data()); return false;
    }
    GlDeleteBuffers(static_cast<GLsizei>(RaySceneBuffers.size()),RaySceneBuffers.data());
    RaySceneBuffers=buffers; UploadedRayScene=FrameRayScene; return true;
}
