// Storage bindings connect decoded streams to the visibility table. Allocation
// and validation precede BeginPass; only Draw records GPU assembly commands.
struct NaniteAssemblySource
{
    NaniteVisibilityGeometryConstants Constants; unsigned GroupsX=0,GroupsY=0; BufferHandle Vertex,Index;
};
std::vector<NaniteAssemblySource> NaniteAssemblySources;
GLuint NaniteAssemblyProgram=0,NaniteAssemblyUniform=0;
void DestroyNaniteVisibilityGeometry()
{ DestroyNaniteInstances();
    NaniteAssemblySources.clear();
    if (NaniteAssemblyProgram && GlDeleteProgram) GlDeleteProgram(NaniteAssemblyProgram);
    if (NaniteAssemblyUniform && GlDeleteBuffers) GlDeleteBuffers(1,&NaniteAssemblyUniform);
    NaniteAssemblyProgram=NaniteAssemblyUniform=0;
}
bool PrepareNaniteVisibilityGeometry(const VisibilityScene& scene)
{
    if (!NaniteGeometrySupported) return Fail("ES native visibility assembly requires compute storage");
    if (!ValidateVisibilityScene(scene,LastError)) return false;
    const auto bytes=scene.TriangleCount()*sizeof(VisibilityTriangle);
    std::vector<NaniteAssemblySource> sources; sources.reserve(scene.NativeMeshes.size());
    for (unsigned i=0;i<scene.NativeMeshes.size();++i)
    {
        const auto& mesh=scene.NativeMeshes[i]; const auto vertex=Buffers.find(mesh.VertexBuffer.Value),index=Buffers.find(mesh.IndexBuffer.Value);
        if (vertex==Buffers.end() || index==Buffers.end() || !vertex->second.NaniteGeometry || !index->second.NaniteGeometry)
            return Fail("ES visibility assembly requires native decoded buffers");
        NaniteAssemblySource source; source.Vertex=mesh.VertexBuffer; source.Index=mesh.IndexBuffer;
        if (!BuildNaniteVisibilityGeometryConstants(scene,i,vertex->second.Description,index->second.Description,source.Constants,source.GroupsX,source.GroupsY,LastError)) return false;
        if (source.GroupsX>unsigned(NaniteMaximumGroupsX) || source.GroupsY>unsigned(NaniteMaximumGroupsY) ||
            vertex->second.Description.SizeInBytes>VisibilityMaximumBytes || index->second.Description.SizeInBytes>VisibilityMaximumBytes)
            return Fail("ES visibility assembly exceeds device storage/dispatch limits");
        sources.push_back(source);
    }
    if (!NaniteAssemblyProgram)
    {
        std::string source; if (!LoadTechniqueShaderSource("OpenGLES","NaniteRaster.glsl",source,LastError)) return false;
        const auto version=source.find('\n'); if (version==std::string::npos) return Fail("ES assembly shader has no version");
        source.insert(version+1,VisibilityShaderDefines()+"#define EVANS_NANITE_VISIBILITY_GEOMETRY 1\n");
        const auto program=CompileComputeProgram(source); if(!program)return false;
        NaniteAssemblyProgram=program;
    }
    if (!AllocateVisibilityGeometry(bytes,nullptr,GL_DYNAMIC_DRAW)) return false;
    GLint uniform=0; GlGetIntegerv(GpuUniformBinding,&uniform);
    if (!NaniteAssemblyUniform) GlGenBuffers(1,&NaniteAssemblyUniform);
    GlBindBuffer(GL_UNIFORM_BUFFER,NaniteAssemblyUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(NaniteVisibilityGeometryConstants),nullptr,GL_DYNAMIC_DRAW);
    GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(uniform));
    if (!CheckGL("Allocate Nanite visibility assembly constants")) return false;
    NaniteAssemblySources=std::move(sources); return true;
}
bool DrawNaniteVisibilityGeometry()
{
    NaniteCopyLocks.reserve(NaniteCopyLocks.size()+NaniteAssemblySources.size()*2);
    for (const auto& source:NaniteAssemblySources) { NaniteCopyLocks.push_back(source.Vertex); NaniteCopyLocks.push_back(source.Index); }
    NaniteFrameBindings restore(*this); GlBindFramebuffer(GL_FRAMEBUFFER,0); GlUseProgram(NaniteAssemblyProgram);
    GlBindBuffer(GL_UNIFORM_BUFFER,NaniteAssemblyUniform); GlBindBufferBase(GL_UNIFORM_BUFFER,0,NaniteAssemblyUniform);
    BindVisibilityGeometry(2);
    for (const auto& source:NaniteAssemblySources)
    {
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,Buffers.at(source.Vertex.Value).Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,Buffers.at(source.Index.Value).Buffer);
        GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(source.Constants),&source.Constants);
        GpuDispatch(source.GroupsX,source.GroupsY,1); ++PendingFrame.Statistics.NaniteGeometryAssemblyCount;
    }
    GpuMemoryBarrier(GpuStorageBarrier); return CheckGL("Assemble native Nanite visibility geometry");
}
