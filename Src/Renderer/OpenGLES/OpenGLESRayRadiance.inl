// Immutable scene radiance storage is shared by diffuse GI and specular
// reflections. Each effect retains independent shaders, constants and targets.
RayRadianceFrame FrameRayRadiance;
Techniques::Settings FrameRayRadianceSettings;
std::shared_ptr<const RaySceneShadingUpload> UploadedRayShading;
std::shared_ptr<const SceneLightCollection> UploadedRayLights;
std::array<GLuint,RayRadianceBufferCount> RayRadianceBuffers{};
GLuint RayRadianceColorAtlas=0;
static constexpr std::size_t RayRadianceAtlasSlot=RaySceneShadingBufferCount-1;
bool RayRadianceSupported=false;

void DestroyRayRadiance()
{
    if (RayRadianceColorAtlas && GlDeleteTextures) GlDeleteTextures(1,&RayRadianceColorAtlas);
    if (GlDeleteBuffers) GlDeleteBuffers(static_cast<GLsizei>(RayRadianceBuffers.size()),RayRadianceBuffers.data());
    RayRadianceBuffers={}; RayRadianceColorAtlas=0; RayRadianceSupported=false;
    UploadedRayShading.reset(); UploadedRayLights.reset(); FrameRayRadiance={};
}

bool EnsureRayRadiance()
{
    const auto& frame=FrameRayRadiance;
    if (!RayRadianceSupported || !frame.Geometry || frame.Geometry!=FrameRayScene || !frame.Shading || !frame.Lights)
        return Fail("OpenGL ES ray radiance requires seven fragment storage blocks and matched scene snapshots");
    if (!EnsureRayScene()) return false;
    const bool materials=frame.Shading!=UploadedRayShading,lights=frame.Lights!=UploadedRayLights;
    if (!materials && !lights) return true;
    if (materials && !ValidateRaySceneShadingUpload(*frame.Geometry,*frame.Shading,LastError)) return false;
    GLint maximumBytes=0;
    GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&maximumBytes);
    if (!CheckGL("Query ray reflection storage limits") || maximumBytes<=0) return false;
    const auto data=GetRayRadianceBufferData(frame);
    std::array<GLuint,RayRadianceBufferCount> replacements{};
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (slot==RayRadianceAtlasSlot || !(slot<RaySceneShadingBufferCount ? materials : lights)) continue;
        if (data[slot].Bytes>static_cast<std::size_t>(maximumBytes))
        {
            GlDeleteBuffers(static_cast<GLsizei>(replacements.size()),replacements.data());
            return Fail("OpenGL ES ray shading exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE");
        }
        GlGenBuffers(1,&replacements[slot]);
        GlBindBuffer(GL_SHADER_STORAGE_BUFFER,replacements[slot]);
        GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(data[slot].Bytes),data[slot].Data,GL_STATIC_DRAW);
    }
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,0);
    if (!CheckGL("Upload ray reflection shading"))
    { GlDeleteBuffers(static_cast<GLsizei>(replacements.size()),replacements.data()); return false; }
    GLuint atlas=0;
    if (materials)
    {
        // Upload complete rows directly from the immutable CPU snapshot. The
        // final partial row never reads padding beyond the source allocation.
        const auto count=std::max<std::size_t>(1,frame.Shading->Texels.size());
        const auto width=std::min<std::size_t>(count,static_cast<std::size_t>(MaximumTextureSize));
        const auto height=width ? (count+width-1)/width : 0;
        if (!height || height>static_cast<std::size_t>(MaximumTextureSize))
        {
            GlDeleteBuffers(static_cast<GLsizei>(replacements.size()),replacements.data());
            return Fail("OpenGL ES reflection color atlas exceeds GL_MAX_TEXTURE_SIZE");
        }
        UnbindSampledTextures(); GlActiveTexture(GL_TEXTURE0); GlGenTextures(1,&atlas);
        GlBindTexture(GL_TEXTURE_2D,atlas); ConfigureDepthTexture();
        GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,static_cast<GLsizei>(width),static_cast<GLsizei>(height),0,GL_RGBA,GL_FLOAT,nullptr);
        const auto* texels=static_cast<const Float4*>(data[RayRadianceAtlasSlot].Data);
        const auto rows=count/width,tail=count%width;
        if (rows) GlTexSubImage2D(GL_TEXTURE_2D,0,0,0,static_cast<GLsizei>(width),static_cast<GLsizei>(rows),GL_RGBA,GL_FLOAT,texels);
        if (tail) GlTexSubImage2D(GL_TEXTURE_2D,0,0,static_cast<GLint>(rows),static_cast<GLsizei>(tail),1,GL_RGBA,GL_FLOAT,texels+rows*width);
        GlBindTexture(GL_TEXTURE_2D,0);
        if (!CheckGL("Upload ray reflection RGBA32F atlas"))
        { GlDeleteTextures(1,&atlas); GlDeleteBuffers(static_cast<GLsizei>(replacements.size()),replacements.data()); return false; }
    }
    for (std::size_t slot=0;slot<replacements.size();++slot) if (replacements[slot])
    { GlDeleteBuffers(1,&RayRadianceBuffers[slot]); RayRadianceBuffers[slot]=replacements[slot]; }
    if (atlas) { GlDeleteTextures(1,&RayRadianceColorAtlas); RayRadianceColorAtlas=atlas; }
    UploadedRayShading=frame.Shading; UploadedRayLights=frame.Lights; return true;
}
