static constexpr GLuint RayShadowTextureUnit=8;
static constexpr GLenum RayShadowRGBA32UI=0x8D70,RayShadowRGBAInteger=0x8D99;
static constexpr GLenum RayShadowActiveTexture=0x84E0,RayShadowTextureBinding=0x8069;
std::shared_ptr<const RayShadowImage> FrameRayShadowImage,CachedRayShadowImage;
GLuint RayShadowTexture=0,RayShadowDummyTexture=0;
std::array<GLuint,13> RayShadowPrograms{};

bool CreateRayShadowTexture(const RayShadowImage& image,GLuint& output)
{
    if (!image.Width || !image.Height || image.Texels.size()!=std::size_t(image.Width)*image.Height)
        return Fail("GLES ray-shadow image has invalid dimensions or payload");
    GLint active=0,texture=0,unpack=0;
    GlGetIntegerv(RayShadowActiveTexture,&active); GlActiveTexture(GL_TEXTURE0+RayShadowTextureUnit);
    GlGetIntegerv(RayShadowTextureBinding,&texture); GlGetIntegerv(TSRPixelUnpackBinding,&unpack);
    GLuint candidate=0; GlGenTextures(1,&candidate); GlBindTexture(GL_TEXTURE_2D,candidate);
    ConfigureDepthTexture();
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlBindBuffer(TSRPixelUnpackBuffer,0);
    GlTexImage2D(GL_TEXTURE_2D,0,RayShadowRGBA32UI,image.Width,image.Height,0,
        RayShadowRGBAInteger,GL_UNSIGNED_INT,image.Texels.data());
    const bool valid=CheckGL("Upload GLES ray-shadow integer image");
    GlBindTexture(GL_TEXTURE_2D,texture); GlBindBuffer(TSRPixelUnpackBuffer,unpack); GlActiveTexture(active);
    if (!valid) { GlDeleteTextures(1,&candidate); return false; }
    output=candidate; return true;
}
bool PrepareRayShadowImage(std::shared_ptr<const RayShadowImage> image)
{
    if (!LumenBindSampler && !LoadGL(LumenBindSampler,"glBindSampler")) return false;
    if (!RayShadowDummyTexture)
    {
        RayShadowImage empty; empty.Width=RayShadowHeaderTexels; empty.Height=1; empty.Texels.resize(RayShadowHeaderTexels);
        if (!CreateRayShadowTexture(empty,RayShadowDummyTexture)) return false;
    }
    if (image && image!=CachedRayShadowImage)
    {
        GLuint candidate=0; if (!CreateRayShadowTexture(*image,candidate)) return false;
        if (RayShadowTexture) GlDeleteTextures(1,&RayShadowTexture);
        RayShadowTexture=candidate; CachedRayShadowImage=image;
    }
    FrameRayShadowImage=std::move(image); return true;
}
void BindRayShadowImage(GLuint program)
{
    const auto location=GlGetUniformLocation(program,"ShadowRayScene");
    if (location<0) return;
    GlActiveTexture(GL_TEXTURE0+RayShadowTextureUnit); LumenBindSampler(RayShadowTextureUnit,0);
    GlBindTexture(GL_TEXTURE_2D,FrameRayShadowImage?RayShadowTexture:RayShadowDummyTexture);
    GlUniform1i(location,RayShadowTextureUnit);
}
GLuint RayShadowProgram(unsigned program,bool weighted=false)
{
    constexpr unsigned SurfacePrograms=5,ProgramCount=8;
    if (program>=ProgramCount || (weighted && program>=SurfacePrograms))
    { Fail("GLES ray-shadow material program is invalid"); return 0; }
    auto& shader=RayShadowPrograms[weighted?ProgramCount+program:program];
    if (!shader)
    {
        const auto definitions=std::string("#define EVANS_RAY_SHADOW_PROGRAM ")+std::to_string(program)+"\n"+
            (weighted?"#define EVANS_WEIGHTED_TRANSPARENCY 1\n":"");
        if (program>=SurfacePrograms) shader=CompileStageProgram("RayTracedShadows.glsl",definitions.c_str(),program==6);
        else
        {
            // Material programs reuse this API's ordinary vertex interface;
            // the ray-shadow file owns the independent fragment implementation.
            std::string vertexSource,fragmentSource;
            const auto vertexFile=std::string(ShadingTechniqueFile(Techniques::ShadingModel(program),false))+".glsl";
            if (!LoadTechniqueShaderSource("OpenGLES",vertexFile.c_str(),vertexSource,LastError) ||
                !LoadTechniqueShaderSource("OpenGLES","RayTracedShadows.glsl",fragmentSource,LastError)) return 0;
            vertexSource.insert(vertexSource.find('\n')+1,"#define EVANS_VERTEX_STAGE 1\n");
            fragmentSource.insert(fragmentSource.find('\n')+1,definitions);
            const auto linked=CompileProgramPair(vertexSource,fragmentSource);
            if(!linked)return 0;
            shader=linked;
        }
    }
    return shader;
}
void DestroyRayShadowImage()
{
    if (RayShadowTexture) GlDeleteTextures(1,&RayShadowTexture);
    if (RayShadowDummyTexture) GlDeleteTextures(1,&RayShadowDummyTexture);
    RayShadowTexture=RayShadowDummyTexture=0;
    for (auto& program:RayShadowPrograms) { if (program) GlDeleteProgram(program); program=0; }
    FrameRayShadowImage.reset(); CachedRayShadowImage.reset();
}
