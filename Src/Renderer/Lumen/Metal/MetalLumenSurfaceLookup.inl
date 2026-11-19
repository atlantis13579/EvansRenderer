ObjCObject LumenSurfaceLookup=nullptr;
void DestroyLumenSurfaceLookup() { ReleaseObject(LumenSurfaceLookup); LumenSurfaceLookup=nullptr; }
bool EnsureLumenSurfaceLookup()
{
    if (LumenSurfaceLookup) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
    struct Compilation
    {
        ObjCObject Options=nullptr,Library=nullptr,Function=nullptr;
        ~Compilation() { ReleaseObject(Function); ReleaseObject(Library); ReleaseObject(Options); }
    } compilation;
    compilation.Options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(compilation.Options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    compilation.Library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),compilation.Options,&error);
    if (!compilation.Library) return Fail("Metal Lumen surface lookup compilation failed: "+ErrorDescription(error));
    compilation.Function=Send<ObjCObject,ObjCObject>(compilation.Library,"newFunctionWithName:",MakeString("lumenLookupSurface"));
    if (!compilation.Function) return Fail("Metal Lumen surface lookup kernel is unavailable");
    LumenSurfaceLookup=Evans::Renderer::CreateCachedMetalComputePipeline(Device,compilation.Function,&error);
    if (!LumenSurfaceLookup) return Fail("Metal Lumen surface lookup pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(LumenSurfaceLookup,"maxTotalThreadsPerThreadgroup")<LumenLookupThreads)
    { DestroyLumenSurfaceLookup(); return Fail("Metal Lumen lookup exceeds the device threadgroup size"); }
    return true;
}
bool LookupLumenSurface(LumenSurfaceHandle handle,const LumenSurfaceLookupUpload& upload,LumenSurfaceLookupResult& output)
{
    const auto found=LumenSurfaces.find(handle.Value); if (found==LumenSurfaces.end()) return Fail("Unknown Lumen surface lookup handle");
    const auto record=found->second; LumenSurfaceLookupFrame frame; std::vector<LumenLookupCard> cards;
    if (!BuildLumenSurfaceLookupFrame(record->Layout,upload,frame,LastError) || !BuildLumenLookupCards(record->Layout,record->Captures,cards,LastError) || !EnsureLumenSurfaceLookup()) return false;
    struct Resources
    {
        ObjCObject Cards=nullptr,Queries=nullptr,Feedback=nullptr;
        ~Resources() { ReleaseObject(Cards); ReleaseObject(Queries); ReleaseObject(Feedback); }
    } resources;
    resources.Cards=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",cards.data(),cards.size()*sizeof(LumenLookupCard),0);
    resources.Queries=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",upload.Queries.data(),upload.Queries.size()*sizeof(LumenSurfaceQuery),0);
    const auto bytes=upload.Queries.size()*sizeof(LumenSurfaceSample); resources.Feedback=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",bytes,0);
    if (!resources.Cards || !resources.Queries || !resources.Feedback || !Send<void*>(resources.Feedback,"contents")) return Fail("Metal Lumen lookup allocation failed");
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal Lumen lookup command buffer is unavailable");
    const auto encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal Lumen lookup encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",LumenSurfaceLookup); Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&frame,sizeof(frame),0);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",resources.Cards,0,1);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",resources.Queries,0,2);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",resources.Feedback,0,3);
    for (NSUInteger plane=0;plane<LumenSurfacePlanes;++plane) Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",record->Images[plane],plane);
    const auto groups=unsigned((upload.Queries.size()+LumenLookupThreads-1)/LumenLookupThreads);
    const MTLSize grid{std::min(groups,LumenLookupColumns),(groups+LumenLookupColumns-1)/LumenLookupColumns,1},threads{LumenLookupThreads,1,1};
    Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",grid,threads); Send<void>(encoder,"endEncoding");
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Lumen lookup submission failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    LumenSurfaceLookupResult result; result.Identity=record->Identity; result.Samples.resize(upload.Queries.size()); std::memcpy(result.Samples.data(),Send<void*>(resources.Feedback,"contents"),bytes);
    if (!ValidateLumenSurfaceLookupResult(record->Identity,upload,result,LastError)) return false;
    output=std::move(result); return true;
}
