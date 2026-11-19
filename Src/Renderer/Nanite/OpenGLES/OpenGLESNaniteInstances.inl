NaniteInstanceWork InstancePlan;
std::vector<std::shared_ptr<NaniteFrameBuffer>> InstanceBuffers;
std::array<GLuint,3> InstanceKernels{};
GLuint InstanceUniform=0;
void(EVANS_GL_CALL* InstanceDispatchIndirect)(GLintptr)=nullptr;
static constexpr GLenum InstanceIndirectBuffer=0x90EE,InstanceIndirectBinding=0x90EF;
void DestroyNaniteInstances()
{
    InstancePlan={}; InstanceBuffers.clear();
    for (auto program:InstanceKernels) if (program && GlDeleteProgram) GlDeleteProgram(program); InstanceKernels={};
    if (InstanceUniform && GlDeleteBuffers) GlDeleteBuffers(1,&InstanceUniform); InstanceUniform=0;
}
bool PrepareNaniteInstances(const VisibilityBufferDesc& frame)
{
    std::vector<NaniteVisibilityGeometryConstants> constants;
    if (frame.GpuInstances) for (const auto& source:NaniteAssemblySources) constants.push_back(source.Constants);
    if (!BuildNaniteInstanceWork(frame,constants,InstancePlan,LastError)) return false;
    InstanceBuffers.clear(); if (InstancePlan.Batches.empty()) return true;
    if (!LoadGL(InstanceDispatchIndirect,"glDispatchComputeIndirect")) return false;
    GLint blocks=0; GlGetIntegerv(TessellationComputeBlocks,&blocks);
    if (blocks<4) return Fail("ES instance assembly needs four compute storage bindings");
    for (const auto& batch:InstancePlan.Batches)
    {
        if (batch.MaximumGroupsX>unsigned(NaniteMaximumGroupsX) || batch.MaximumGroupsY>unsigned(NaniteMaximumGroupsY) ||
            (batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads>unsigned(NaniteMaximumGroupsX))
            return Fail("ES instance schedule exceeds indirect dispatch limits");
        std::shared_ptr<NaniteFrameBuffer> work;
        if (!CreateNaniteFrameBuffer(unsigned(batch.Upload.size()),batch.Upload.data(),true,work)) return false;
        InstanceBuffers.push_back(std::move(work));
    }
    if (!InstanceUniform)
    {
        GLint previous=0; GlGetIntegerv(GpuUniformBinding,&previous); GlGenBuffers(1,&InstanceUniform);
        GlBindBuffer(GL_UNIFORM_BUFFER,InstanceUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(ShaderUInt4),nullptr,GL_DYNAMIC_DRAW); GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(previous));
        if (!CheckGL("Allocate instance schedule constants")) return false;
    }
    if (std::all_of(InstanceKernels.begin(),InstanceKernels.end(),[](auto program){return program!=0;})) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","NaniteInstances.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("ES instance shader has no version");
    for (unsigned pass=0;pass<InstanceKernels.size();++pass) if (!InstanceKernels[pass])
    {
        auto text=source; text.insert(version+1,VisibilityShaderDefines()+"#define EVANS_INSTANCE_PASS "+std::to_string(pass)+"\n");
        const auto program=CompileComputeProgram(text); if(!program)return false;
        InstanceKernels[pass]=program;
    }
    return true;
}
bool DrawNaniteInstances()
{
    NaniteFrameBindings restore(*this); GLint previousIndirect=0; GlGetIntegerv(InstanceIndirectBinding,&previousIndirect);
    GlBindFramebuffer(GL_FRAMEBUFFER,0); GlBindBuffer(GL_UNIFORM_BUFFER,InstanceUniform); GlBindBufferBase(GL_UNIFORM_BUFFER,0,InstanceUniform);
    BindVisibilityGeometry(2);
    for (unsigned family=0;family<InstancePlan.Batches.size();++family)
    {
        const auto& batch=InstancePlan.Batches[family]; const auto buffer=InstanceBuffers[family]->Buffer;
        NaniteCopyLocks.push_back(batch.Vertex); NaniteCopyLocks.push_back(batch.Index);
        GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(batch.Parameters),&batch.Parameters);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,NaniteRasterDecisions->Buffer);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,Buffers.at(batch.Index.Value).Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,buffer);
        GlUseProgram(InstanceKernels[0]); GpuDispatch((batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads,1,1); GpuMemoryBarrier(GpuStorageBarrier);
        GlUseProgram(InstanceKernels[1]); GpuDispatch(1,1,1); GpuMemoryBarrier(GpuStorageBarrier|GpuCommandBarrier);
        GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,Buffers.at(batch.Vertex.Value).Buffer); GlBindBuffer(InstanceIndirectBuffer,buffer);
        GlUseProgram(InstanceKernels[2]); InstanceDispatchIndirect(0); GpuMemoryBarrier(GpuStorageBarrier);
        ++PendingFrame.Statistics.NaniteGeometryAssemblyCount; PendingFrame.Statistics.NaniteInstanceDispatchCount+=unsigned(InstanceKernels.size());
    }
    GlBindBuffer(InstanceIndirectBuffer,GLuint(previousIndirect)); return CheckGL("Schedule and assemble Nanite instances");
}
