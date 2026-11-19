struct LumenLightingRecord
{
    std::vector<unsigned> CardInstances; LumenLightingInfo Info; ObjCObject Buffer=nullptr;
    std::shared_ptr<LumenSurfaceRecord> Surface; std::shared_ptr<LumenDistanceRecord> Distance;
    ~LumenLightingRecord() { ReleaseObject(Buffer); }
};
std::unordered_map<std::uint64_t,std::shared_ptr<LumenLightingRecord>> LumenLightingCaches;
ObjCObject LumenSurfaceLighting=nullptr,LumenSurfaceRadiosity=nullptr;
void DestroyLumenLightingCaches()
{ DestroyLumenLightingAsync(); LumenLightingCaches.clear(); ReleaseObject(LumenSurfaceLighting); LumenSurfaceLighting=nullptr;
    ReleaseObject(LumenSurfaceRadiosity); LumenSurfaceRadiosity=nullptr; }
bool EnsureLumenSurfaceLighting(bool indirect=false,bool hardware=false)
{
    if (hardware) return EnsureLumenHardwarePipeline(indirect?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct);
    auto& pipeline=indirect?LumenSurfaceRadiosity:LumenSurfaceLighting;
    if (pipeline) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    struct Compilation
    {
        ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
        ~Compilation() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); }
    } compilation;
    compilation.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(compilation.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    compilation.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),compilation.Options,&error);
    if (!compilation.Library) return Fail("Metal Lumen lighting compilation failed: "+ErrorDescription(error));
    compilation.Function=Send<ObjCObject,ObjCObject>(compilation.Library,"newFunctionWithName:",MakeString(indirect?"lumenBounceSurface":"lumenLightSurface"));
    if (!compilation.Function) return Fail("Metal Lumen lighting kernel is unavailable");
    pipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,compilation.Function,&error);
    if (!pipeline) return Fail("Metal Lumen lighting pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(pipeline,"maxTotalThreadsPerThreadgroup")<LumenLightingThreads)
    { DestroyLumenLightingCaches(); return Fail("Metal Lumen lighting exceeds the device threadgroup size"); }
    return true;
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
    const auto* contents=Send<void*>(record.Buffer,"contents"); if (!contents) return Fail("Metal Lumen lighting buffer is unavailable");
    std::vector<LumenLightingTexel> texels(std::size_t(record.Info.Layout.Width)*record.Info.Layout.Height);
    std::memcpy(texels.data(),contents,texels.size()*sizeof(LumenLightingTexel));
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
    if (!BuildLumenSurfaceLightingInput(material->second->Identity,material->second->Layout,material->second->Captures,geometry->second->Info,upload,input,LastError) || !EnsureLumenSurfaceLighting(bool(previous),bool(geometry->second->Hardware))) return false;
    auto record=std::make_shared<LumenLightingRecord>(); record->Surface=material->second; record->Distance=geometry->second;
    record->CardInstances=upload.CardInstances;
    record->Info={{record->Surface->Identity,record->Distance->Info.Identity,upload.LightingRevision},record->Surface->Layout,input.Frame.Work.Y};
    record->Info.HairSky={input.Frame.Trace.HairSky.X,input.Frame.Trace.HairSky.Y,input.Frame.Trace.HairSky.Z};
    if (previous) { record->Info.IndirectSamples=upload.IndirectRays; record->Info.Bounce=previous->Info.Bounce+1; }
    struct InputOwner { ObjCObject Buffer=nullptr; ~InputOwner() { ReleaseObject(Buffer); } } resources;
    resources.Buffer=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",input.Words.data(),input.Words.size()*sizeof(std::uint32_t),0);
    record->Buffer=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",NSUInteger(input.Frame.Trace.Counts.X)*sizeof(LumenLightingTexel),0);
    if (!resources.Buffer || !record->Buffer || !Send<void*>(record->Buffer,"contents")) return Fail("Metal Lumen lighting allocation failed");
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen lighting command buffer is unavailable");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen lighting encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",record->Distance->Hardware?LumenHardwarePipelines[unsigned(previous?LumenHardwareProgram::Bounce:LumenHardwareProgram::Direct)]:previous?LumenSurfaceRadiosity:LumenSurfaceLighting);
    BindLumenHardware(encoder,*record->Distance); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&input.Frame,sizeof(input.Frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Distance->Buffer,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",resources.Buffer,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",record->Buffer,0,3);
    if (previous) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",previous->Buffer,0,4);
    for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",record->Surface->Images[plane],plane);
    const auto groups=LumenLightingGroups(input.Frame);
    const MTLSize grid{std::min(groups,LumenLightingColumns),(groups+LumenLightingColumns-1)/LumenLightingColumns,1},threads{LumenLightingThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Lumen lighting submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    LumenSurfaceLightingResult result; result.Info=record->Info;
    if (upload.Readback && !ReadLumenLightingRecord(*record,result.Texels)) return false;
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
