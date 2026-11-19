// ES storage/command barriers publish the compact list and its draw arguments.
NaniteMaterialWork MaterialPlan;
GLuint MaterialMap=0,MaterialWork=0,MaterialUniform=0,MaterialTileImage=0;
unsigned MaterialImageWidth=0,MaterialImageHeight=0;
std::array<GLuint,4> MaterialKernels{};
void(EVANS_GL_CALL* MaterialDrawIndirect)(GLenum,const void*)=nullptr;
void(EVANS_GL_CALL* MaterialBindSampler)(GLuint,GLuint)=nullptr;
void(EVANS_GL_CALL* MaterialTextureStorage)(GLenum,GLsizei,GLenum,GLsizei,GLsizei)=nullptr;
void(EVANS_GL_CALL* MaterialBindImage)(GLuint,GLuint,GLint,GLboolean,GLint,GLenum,GLenum)=nullptr;
void DestroyNaniteMaterials()
{
    for (auto buffer:{MaterialMap,MaterialWork,MaterialUniform}) if (buffer && GlDeleteBuffers) GlDeleteBuffers(1,&buffer);
    for (auto program:MaterialKernels) if (program && GlDeleteProgram) GlDeleteProgram(program);
    if (MaterialTileImage && GlDeleteTextures) GlDeleteTextures(1,&MaterialTileImage);
    MaterialImageWidth=MaterialImageHeight=0;
    MaterialMap=MaterialWork=MaterialUniform=MaterialTileImage=0; MaterialKernels={}; MaterialPlan={};
}
bool PrepareNaniteMaterials(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteMaterialWork(frame,CreateInfo.Width,CreateInfo.Height,MaterialPlan,LastError)) return false;
    if (!MaterialPlan.Active()) return true;
    if (!LoadGL(MaterialTextureStorage,"glTexStorage2D")) return false;
    GLint storage=0,uniform=0; GlGetIntegerv(VisibilityStorageBinding,&storage); GlGetIntegerv(VisibilityUniformBinding,&uniform);
    if (!MaterialMap) GlGenBuffers(1,&MaterialMap);
    if (!MaterialWork) GlGenBuffers(1,&MaterialWork);
    if (MaterialPlan.Active())
    {
        // The compact list lives in the integer image; raw storage ends after temporary entries.
        const auto bytes=std::size_t(MaterialPlan.Constants.Storage.Y)*sizeof(unsigned);
        const unsigned imageWidth=unsigned(std::min(CreateInfo.Width,MaximumTextureSize));
        const auto imageWords=MaterialPlan.Representatives.size()+MaterialPlan.Constants.Storage.X;
        const unsigned imageHeight=unsigned((imageWords+imageWidth-1)/imageWidth);
        const auto imageBytes=std::size_t(imageWidth)*imageHeight*sizeof(unsigned);
        MaterialPlan.Constants.Storage.W=imageWidth;
        if (bytes>VisibilityMaximumBytes || MaterialPlan.MaterialBins.size()*sizeof(unsigned)>VisibilityMaximumBytes ||
            imageHeight>unsigned(MaximumTextureSize) || bytes+imageBytes+MaterialPlan.MaterialBins.size()*sizeof(unsigned)>NaniteMaterialMaximumBytes)
        { GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(storage)); return Fail("ES material schedule exceeds storage-block capacity"); }
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,MaterialMap); GlBufferData(GL_SHADER_STORAGE_BUFFER,MaterialPlan.MaterialBins.size()*sizeof(unsigned),MaterialPlan.MaterialBins.data(),GL_STATIC_DRAW);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,MaterialWork); GlBufferData(GL_SHADER_STORAGE_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW);
        if (MaterialImageWidth!=imageWidth || MaterialImageHeight!=imageHeight)
        {
            GLint texture=0; GlGetIntegerv(VisibilityTexture2DBinding,&texture);
            if (MaterialTileImage) GlDeleteTextures(1,&MaterialTileImage); GlGenTextures(1,&MaterialTileImage);
            GlBindTexture(GL_TEXTURE_2D,MaterialTileImage); ConfigureDepthTexture();
            MaterialTextureStorage(GL_TEXTURE_2D,1,VisibilityR32UI,imageWidth,imageHeight);
            GlBindTexture(GL_TEXTURE_2D,GLuint(texture)); MaterialImageWidth=imageWidth; MaterialImageHeight=imageHeight;
        }
        if (!MaterialUniform) GlGenBuffers(1,&MaterialUniform);
        GlBindBuffer(GL_UNIFORM_BUFFER,MaterialUniform); GlBufferData(GL_UNIFORM_BUFFER,sizeof(NaniteMaterialConstants),&MaterialPlan.Constants,GL_DYNAMIC_DRAW);
    }
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,GLuint(storage)); GlBindBuffer(GL_UNIFORM_BUFFER,GLuint(uniform));
    if (!CheckGL("Prepare material schedule storage")) { MaterialImageWidth=MaterialImageHeight=0; return false; }
    if (!LoadGL(MaterialBindImage,"glBindImageTexture") || !LoadGL(MaterialBindSampler,"glBindSampler") || !LoadGL(MaterialDrawIndirect,"glDrawArraysIndirect") || !LoadGL(GpuDispatch,"glDispatchCompute") || !LoadGL(GpuMemoryBarrier,"glMemoryBarrier")) return false;
    if (!VisibilityPrograms[3]) VisibilityPrograms[3]=CompileStageProgram("VisibilityBuffer.glsl",(VisibilityShaderDefines()+"#define EVANS_VISIBILITY_RESOLVE 1\n#define EVANS_MATERIAL_TILES 1\n").c_str());
    if (!VisibilityPrograms[3]) return false;
    GLint invocations=0,width=0,blocks=0,maxX=0,maxY=0;
    GlGetIntegerv(TessellationInvocations,&invocations); GlGetIntegerv(TessellationComputeBlocks,&blocks);
    VisibilityGetIndexed(TessellationGroupSize,0,&width); VisibilityGetIndexed(TessellationGroupCount,0,&maxX); VisibilityGetIndexed(TessellationGroupCount,1,&maxY);
    const auto& c=MaterialPlan.Constants;
    if (invocations<int(NaniteMaterialThreads) || width<int(NaniteMaterialThreads) || blocks<3 || c.Layout.Y>unsigned(maxX) || c.Layout.Z>unsigned(maxY) ||
        (c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads>unsigned(maxX)) return Fail("ES material schedule exceeds compute capacity");
    if (std::all_of(MaterialKernels.begin(),MaterialKernels.end(),[](auto kernel){return kernel!=0;})) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","NaniteMaterials.glsl",source,LastError)) return false;
    for (unsigned pass=0;pass<MaterialKernels.size();++pass) if (!MaterialKernels[pass])
    {
        auto text=source; text.insert(text.find('\n')+1,VisibilityShaderDefines()+"#define EVANS_MATERIAL_PASS "+std::to_string(pass)+"\n");
        const auto program=CompileComputeProgram(text); if(!program)return false;
        MaterialKernels[pass]=program;
    }
    return true;
}
bool ClassifyNaniteMaterials()
{
    if (!MaterialPlan.Active()) return true;
    GlBindFramebuffer(GL_FRAMEBUFFER,0); UnbindSampledTextures();
    constexpr GLenum ImageName=0x8F3A,ImageLevel=0x8F3B,ImageLayered=0x8F3C,ImageLayer=0x8F3D,ImageAccess=0x8F3E,ImageFormat=0x906E,WriteOnly=0x88B9;
    const GLenum fields[]{ImageName,ImageLevel,ImageLayered,ImageLayer,ImageAccess,ImageFormat}; GLint priorImage[6]{};
    for (unsigned field=0;field<6;++field) VisibilityGetIndexed(fields[field],0,&priorImage[field]);
    MaterialBindImage(0,MaterialTileImage,0,GL_FALSE_VALUE,0,WriteOnly,VisibilityR32UI);
    if (!CheckGL("Bind material list image"))
    {
        MaterialBindImage(0,GLuint(priorImage[0]),priorImage[1],GLboolean(priorImage[2]),priorImage[3],GLenum(priorImage[4]),GLenum(priorImage[5]));
        return false;
    }
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,MaterialUniform);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,MaterialMap);
    BindVisibilityGeometry(1);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,MaterialWork);
    GlActiveTexture(GL_TEXTURE0); GLint sampler=0; GlGetIntegerv(NaniteSamplerBinding,&sampler); MaterialBindSampler(0,0);
    GlBindTexture(GL_TEXTURE_2D,VisibilityImages[0]);
    GpuMemoryBarrier(NaniteFramebufferBarrier|NaniteTextureFetchBarrier|GpuStorageBarrier);
    for (unsigned pass=0;pass<MaterialKernels.size();++pass)
    {
        const auto& c=MaterialPlan.Constants;
        const unsigned x=pass==0?(c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads:pass==2?1:c.Layout.Y;
        const unsigned y=pass==0 || pass==2?1:c.Layout.Z;
        GlUseProgram(MaterialKernels[pass]); GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier);
    }
    constexpr GLbitfield ImageBarrier=0x00000020;
    GpuMemoryBarrier(GpuStorageBarrier|GpuCommandBarrier|ImageBarrier|NaniteTextureFetchBarrier);
    MaterialBindImage(0,GLuint(priorImage[0]),priorImage[1],GLboolean(priorImage[2]),priorImage[3],GLenum(priorImage[4]),GLenum(priorImage[5]));
    MaterialBindSampler(0,GLuint(sampler)); UnbindSampledTextures();
    auto& stats=PendingFrame.Statistics; stats.NaniteMaterialDispatchCount+=unsigned(MaterialKernels.size());
    stats.NaniteMaterialBinCount+=unsigned(MaterialPlan.Representatives.size()); stats.VisibilityMaterialDrawCount+=unsigned(MaterialPlan.Representatives.size());
    stats.IndirectDrawCallCount+=unsigned(MaterialPlan.Representatives.size()); return CheckGL("Classify compact material tiles");
}
