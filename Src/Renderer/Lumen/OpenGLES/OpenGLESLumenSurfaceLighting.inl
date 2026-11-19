struct LumenLightingRecord
{
    Impl& Owner; std::vector<unsigned> CardInstances; LumenLightingInfo Info; GLuint Buffer=0;
    std::shared_ptr<LumenSurfaceRecord> Surface; std::shared_ptr<LumenDistanceRecord> Distance;
    explicit LumenLightingRecord(Impl& owner):Owner(owner) {}
    ~LumenLightingRecord() { if (Buffer) Owner.GlDeleteBuffers(1,&Buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenLightingRecord>> LumenLightingCaches;
GLuint LumenSurfaceLighting=0,LumenSurfaceRadiosity=0;
void DestroyLumenLightingCaches()
{ DestroyLumenLightingAsync(); LumenLightingCaches.clear(); if (LumenSurfaceLighting) GlDeleteProgram(LumenSurfaceLighting); LumenSurfaceLighting=0;
    if (LumenSurfaceRadiosity) GlDeleteProgram(LumenSurfaceRadiosity); LumenSurfaceRadiosity=0; }
bool EnsureLumenSurfaceLighting(bool indirect=false)
{
    if (!EnsureLumenSurface() || !EnsureLumenDistanceTrace()) return false;
    GLint bindings=0; GlGetIntegerv(DistanceMaxComputeStorage,&bindings);
    if (indirect && bindings<4) return Fail("Lumen radiosity requires four compute storage bindings");
    auto& pipeline=indirect?LumenSurfaceRadiosity:LumenSurfaceLighting;
    if (pipeline) return true;
    std::string source; if (!LoadTechniqueShaderSource("OpenGLES","Lumen.glsl",source,LastError)) return false;
    const auto version=source.find('\n'); if (version==std::string::npos) return Fail("Lumen lighting shader has no version directive");
    source.insert(version+1,indirect?"#define EVANS_LUMEN_LIGHTING_STAGE 1\n#define EVANS_LUMEN_RADIOSITY 1\n":"#define EVANS_LUMEN_LIGHTING_STAGE 1\n"); const auto program=CompileComputeProgram(source); if(!program)return false;
    pipeline=program; return true;
}
bool SelectLumenPreviousLighting(const LumenSurfaceLightingUpload& upload,const std::shared_ptr<LumenSurfaceRecord>& surface,
    const std::shared_ptr<LumenDistanceRecord>& distance,std::shared_ptr<LumenLightingRecord>& output)
{
    if (!upload.PreviousLighting) return true;
    const auto found=LumenLightingCaches.find(upload.PreviousLighting.Value);
    if (found==LumenLightingCaches.end() || found->second->Surface!=surface || found->second->Distance!=distance || found->second->CardInstances!=upload.CardInstances)
        return Fail("Lumen radiosity source is expired or belongs to different native inputs");
    if (!ValidateLumenRadiositySource(surface->Identity,distance->Info.Identity,upload,found->second->Info,LastError)) return false;
    output=found->second; return true;
}
bool ReadLumenLightingRecord(const LumenLightingRecord& record,std::vector<LumenLightingTexel>& output)
{
    const auto bytes=std::size_t(record.Info.Layout.Width)*record.Info.Layout.Height*sizeof(LumenLightingTexel);
    struct Binding
    {
        Impl& Owner; GLint Previous=0;
        explicit Binding(Impl& owner):Owner(owner) { Owner.GlGetIntegerv(GpuStorageBinding,&Previous); }
        ~Binding() { Owner.GlBindBuffer(GL_SHADER_STORAGE_BUFFER,Previous); }
    } binding(*this);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record.Buffer);
    std::vector<LumenLightingTexel> texels(bytes/sizeof(LumenLightingTexel));
    const auto* mapped=LumenMapBuffer(GL_SHADER_STORAGE_BUFFER,0,GLsizeiptr(bytes),LumenMapRead);
    if (!mapped) return Fail("Map Lumen lighting readback failed");
    std::memcpy(texels.data(),mapped,bytes); const auto valid=LumenUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!valid || !CheckGL("Read Lumen lighting")) return Fail("Lumen lighting readback became invalid");
    if (!ValidateLumenLightingTexels(record.Info,texels,LastError)) return false;
    output=std::move(texels); return true;
}
bool RelightLumenSurface(LumenSurfaceHandle surface,LumenDistanceSceneHandle distance,const LumenSurfaceLightingUpload& upload,LumenSurfaceLightingResult& output)
{
    const auto material=LumenSurfaces.find(surface.Value); const auto geometry=LumenDistanceScenes.find(distance.Value);
    if (material==LumenSurfaces.end() || geometry==LumenDistanceScenes.end()) return Fail("Lumen relighting requires live surface and distance handles");
    std::shared_ptr<LumenLightingRecord> previous;
    if (!SelectLumenPreviousLighting(upload,material->second,geometry->second,previous)) return false;
    LumenSurfaceLightingInput input;
    if (!BuildLumenSurfaceLightingInput(material->second->Identity,material->second->Layout,material->second->Captures,geometry->second->Info,upload,input,LastError) || !EnsureLumenSurfaceLighting(bool(previous))) return false;
    const auto bytes=std::size_t(input.Frame.Trace.Counts.X)*sizeof(LumenLightingTexel),inputBytes=input.Words.size()*sizeof(std::uint32_t);
    std::int64_t maximum=0; DistanceGetInteger64(DistanceMaxStorageSize,&maximum);
    if (maximum<=0 || std::max(bytes,inputBytes)>std::uint64_t(maximum)) return Fail("Lumen lighting exceeds ES storage limits");
    const auto groups=LumenLightingGroups(input.Frame);
    const unsigned x=std::min(groups,LumenLightingColumns),y=(groups+LumenLightingColumns-1)/LumenLightingColumns;
    GLint maxX=0,maxY=0; GpuGetIntegerIndexed(DistanceMaxGroups,0,&maxX); GpuGetIntegerIndexed(DistanceMaxGroups,1,&maxY);
    if (x>unsigned(maxX) || y>unsigned(maxY)) return Fail("Lumen lighting exceeds ES dispatch limits");
    auto record=std::make_shared<LumenLightingRecord>(*this); record->Surface=material->second; record->Distance=geometry->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{record->Surface->Identity,record->Distance->Info.Identity,upload.LightingRevision},record->Surface->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=previous->Info.Bounce+1; }
    struct BuffersOwner { Impl& Owner; std::array<GLuint,2> Values{}; ~BuffersOwner() { Owner.GlDeleteBuffers(2,Values.data()); } } buffers{*this};
    LumenCaptureState state(*this); LumenLookupStorageState storageState(*this); GlGenBuffers(2,buffers.Values.data());
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Values[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(input.Frame),&input.Frame,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Values[0]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers.Values[1]); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(inputBytes),input.Words.data(),GL_STATIC_DRAW);
    GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,record->Distance->Buffer); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,buffers.Values[1]);
    GlGenBuffers(1,&record->Buffer); GlBindBuffer(GL_SHADER_STORAGE_BUFFER,record->Buffer); GlBufferData(GL_SHADER_STORAGE_BUFFER,GLsizeiptr(bytes),nullptr,GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,record->Buffer);
    if (previous) GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,previous->Buffer);
    const auto pipeline=previous?LumenSurfaceRadiosity:LumenSurfaceLighting;
    GlUseProgram(pipeline); const char* names[]{"LightingAlbedo","LightingNormal","LightingEmission"};
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        GlActiveTexture(GL_TEXTURE0+plane); LumenBindSampler(plane,0); GlBindTexture(GL_TEXTURE_2D,record->Surface->Images[plane]);
        GlUniform1i(GlGetUniformLocation(pipeline,names[plane]),GLint(plane));
    }
    if (!CheckGL("Bind Lumen surface lighting")) return false;
    GpuDispatch(x,y,1); GpuMemoryBarrier(GpuStorageBarrier|LumenBufferBarrier);
    if (!CheckGL("Dispatch Lumen surface lighting")) return false;
    LumenSurfaceLightingResult result; result.Info=record->Info;
    if (upload.Readback) { if (!ReadLumenLightingRecord(*record,result.Texels)) return false; }
    else { LumenFinish(); if (!CheckGL("Complete Lumen surface lighting")) return false; }
    result.Handle=NewLumenLightingHandle(); if (!result.Handle) return Fail("Lumen lighting handle space exhausted");
    LumenLightingCaches.emplace(result.Handle.Value,std::move(record)); output=std::move(result); return true;
}
bool ReadLumenLighting(LumenLightingHandle handle,std::vector<LumenLightingTexel>& output)
{
    const auto found=LumenLightingCaches.find(handle.Value); if (found==LumenLightingCaches.end()) return Fail("Unknown Lumen lighting cache");
    return ReadLumenLightingRecord(*found->second,output);
}
bool DestroyLumenLighting(LumenLightingHandle handle)
{ return LumenLightingCaches.erase(handle.Value)!=0 || Fail("Unknown Lumen lighting cache"); }
