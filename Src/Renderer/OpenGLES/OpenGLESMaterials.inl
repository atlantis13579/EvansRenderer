std::array<GLuint,MaterialProgramCount> MaterialPrograms{};
GLuint MaterialProgram(MaterialTechnique technique,unsigned model,bool weighted,bool geometry)
{
    if (unsigned(technique)>=MaterialTechniqueCount || model>=MaterialSurfaceModels) { Fail("Invalid GLES material program"); return 0; }
    auto& program=MaterialPrograms[MaterialProgramIndex(technique,model,weighted,geometry)];
    if (program) return program;
    const auto filename=std::string(MaterialTechniqueFile(technique))+".glsl";
    const auto definitions=std::string("#define EVANS_MATERIAL_PROGRAM ")+std::to_string(model)+"\n"+
        (geometry?"#define EVANS_MATERIAL_GBUFFER 1\n":"")+(weighted?"#define EVANS_WEIGHTED_TRANSPARENCY 1\n":"");
    if (geometry) return program=CompileStageProgram(filename.c_str(),definitions.c_str());
    std::string vertexSource,fragmentSource;
    const auto vertexFile=std::string(ShadingTechniqueFile(Techniques::ShadingModel(model),false))+".glsl";
    if (!LoadTechniqueShaderSource("OpenGLES",vertexFile.c_str(),vertexSource,LastError) ||
        !LoadTechniqueShaderSource("OpenGLES",filename.c_str(),fragmentSource,LastError)) return 0;
    vertexSource.insert(vertexSource.find('\n')+1,"#define EVANS_VERTEX_STAGE 1\n");
    fragmentSource.insert(fragmentSource.find('\n')+1,definitions);
    const auto linked=CompileProgramPair(vertexSource,fragmentSource);
    if(!linked)return 0;
    return program=linked;
}
void DestroyMaterials()
{ for (auto& program:MaterialPrograms) { if (program) GlDeleteProgram(program); program=0; } }
