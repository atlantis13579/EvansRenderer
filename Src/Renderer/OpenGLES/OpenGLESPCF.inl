std::array<GLuint,13> PCFPrograms{};
GLuint PCFProgram(unsigned program,bool weighted=false)
{
    constexpr unsigned SurfacePrograms=5,ProgramCount=8;
    if (program>=ProgramCount || (weighted && program>=SurfacePrograms))
    { Fail("GLES PCF material program is invalid"); return 0; }
    auto& shader=PCFPrograms[weighted?ProgramCount+program:program];
    if (!shader)
    {
        const auto definitions=std::string("#define EVANS_PCF_PROGRAM ")+std::to_string(program)+"\n"+
            (weighted?"#define EVANS_WEIGHTED_TRANSPARENCY 1\n":"");
        if (program>=SurfacePrograms) shader=CompileStageProgram("PCF.glsl",definitions.c_str(),program==6);
        else
        {
            // Material programs reuse this API's ordinary vertex interface;
            // the PCF file owns the independent fragment implementation.
            std::string vertexSource,fragmentSource;
            const auto vertexFile=std::string(ShadingTechniqueFile(Techniques::ShadingModel(program),false))+".glsl";
            if (!LoadTechniqueShaderSource("OpenGLES",vertexFile.c_str(),vertexSource,LastError) ||
                !LoadTechniqueShaderSource("OpenGLES","PCF.glsl",fragmentSource,LastError)) return 0;
            vertexSource.insert(vertexSource.find('\n')+1,"#define EVANS_VERTEX_STAGE 1\n");
            fragmentSource.insert(fragmentSource.find('\n')+1,definitions);
            const auto linked=CompileProgramPair(vertexSource,fragmentSource);
            if(!linked)return 0;
            shader=linked;
        }
    }
    return shader;
}

void BindPCF(GLuint program,bool enabled)
{
    const auto constants=BuildPCFConstants(FrameShadowFilter,enabled);
    const auto& kernel=constants.Kernel;
    GlUniform4f(GlGetUniformLocation(program,"PcfKernel"),kernel.X,kernel.Y,kernel.Z,kernel.W);
}
void DestroyPCF()
{
    for (auto& program:PCFPrograms) { if (program) GlDeleteProgram(program); program=0; }
}
