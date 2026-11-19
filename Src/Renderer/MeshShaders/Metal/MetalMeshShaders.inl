// Mesh shaders and object amplification use tracked Metal buffers across the
// existing compute/render encoder boundaries. No CPU reads of Nanite bin sizes.
ObjCObject MeshLibrary=nullptr;
bool MeshNativeBarycentrics=false;
std::array<ObjCObject,4> MeshPipelines{}; // Direct, Nanite depth/owner, ordinary object+mesh.
static constexpr NSUInteger MeshesPerMetalObject=1024,MetalMeshPayloadBytes=16;
bool SupportsMeshShaders() const
{
    if (!Device || !VisibilitySupported || !GetClass("MTLMeshRenderPipelineDescriptor")) return false;
    const auto family=Runtime().RegisterSelector("supportsFamily:");
    const auto create=Runtime().RegisterSelector("newRenderPipelineStateWithMeshDescriptor:options:reflection:error:");
    return Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",family) &&
        Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",create) &&
        (Send<ObjCBool,NSUInteger>(Device,"supportsFamily:",GPUFamilyApple7) || Send<ObjCBool,NSUInteger>(Device,"supportsFamily:",GPUFamilyMac2));
}
void DestroyMeshShaders()
{
    for (auto pipeline:MeshPipelines) ReleaseObject(pipeline); MeshPipelines={};
    ReleaseObject(MeshLibrary); MeshLibrary=nullptr; MeshNativeBarycentrics=false;
}
ObjCObject CreateMeshPipeline(bool nanite,unsigned phase,bool task=false)
{
    const char* fragmentName=nanite?(phase?"EvansHardwareOwner":"EvansHardwareDepth"):
        MeshNativeBarycentrics?"EvansVisibilityBaryFragment":"EvansVisibilityMeshFragment";
    auto mesh=Send<ObjCObject,ObjCObject>(MeshLibrary,"newFunctionWithName:",MakeString(nanite?"EvansNaniteMesh":task?"EvansVisibilityMesh":"EvansVisibilityDirectMesh"));
    auto fragmentLibrary=nanite?HardwareLibrary:MeshNativeBarycentrics?MeshLibrary:VisibilityLibrary;
    auto fragment=Send<ObjCObject,ObjCObject>(fragmentLibrary,"newFunctionWithName:",MakeString(fragmentName));
    auto object=nanite || task?Send<ObjCObject,ObjCObject>(MeshLibrary,"newFunctionWithName:",MakeString(nanite?"EvansNaniteObject":"EvansVisibilityObject")):nullptr;
    if (!mesh || !fragment || ((nanite || task) && !object))
    { ReleaseObject(mesh); ReleaseObject(fragment); ReleaseObject(object); Fail("Metal mesh pipeline entry point is missing"); return nullptr; }
    auto descriptor=Send<ObjCObject>(GetClass("MTLMeshRenderPipelineDescriptor"),"new");
    Send<void,ObjCObject>(descriptor,"setMeshFunction:",mesh); Send<void,ObjCObject>(descriptor,"setFragmentFunction:",fragment);
    Send<void,NSUInteger>(descriptor,"setMaxTotalThreadsPerMeshThreadgroup:",MeshShaderThreads);
    if (object)
    {
        Send<void,ObjCObject>(descriptor,"setObjectFunction:",object);
        Send<void,NSUInteger>(descriptor,"setMaxTotalThreadsPerObjectThreadgroup:",nanite?1:MeshShaderThreads);
        Send<void,NSUInteger>(descriptor,"setPayloadMemoryLength:",nanite?MetalMeshPayloadBytes:MeshShaderPayloadBytes);
    }
    Send<void,NSUInteger>(descriptor,"setRasterSampleCount:",1);
    Send<void,NSUInteger>(descriptor,"setDepthAttachmentPixelFormat:",PixelFormatDepth32Float);
    auto attachments=Send<ObjCObject>(descriptor,"colorAttachments");
    for (NSUInteger i=0;i<2;++i)
    {
        auto attachment=Send<ObjCObject,NSUInteger>(attachments,"objectAtIndexedSubscript:",i);
        Send<void,NSUInteger>(attachment,"setPixelFormat:",i?PixelFormatRG32Float:VisibilityIntegerFormat);
        Send<void,NSUInteger>(attachment,"setWriteMask:",nanite?0:ColorWriteMaskAll);
    }
    ObjCObject error=nullptr;
    auto pipeline=Evans::Renderer::CreateCachedMetalMeshPipeline(Device,descriptor,&error);
    ReleaseObject(descriptor); ReleaseObject(mesh); ReleaseObject(fragment); ReleaseObject(object);
    if (!pipeline) { Fail("Create Metal mesh pipeline: "+ErrorDescription(error)); return nullptr; }
    if (Send<NSUInteger>(pipeline,"maxTotalThreadsPerMeshThreadgroup")<MeshShaderThreads)
    { ReleaseObject(pipeline); Fail("Metal mesh pipeline cannot execute the required workgroup"); return nullptr; }
    return pipeline;
}
bool PrepareMeshShaders(const VisibilityBufferDesc& frame)
{
    if (!frame.MeshShaders) return true;
    if (!SupportsMeshShaders()) return Fail("Metal mesh stages require Metal 3 and Apple7 or Mac2");
    if (frame.NaniteRaster && !HardwarePlan.Active()) return true;
    if (!MeshLibrary)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","MeshShaders.metal",source,LastError)) return false;
        const auto barycentric=Runtime().RegisterSelector("supportsShaderBarycentricCoordinates");
        MeshNativeBarycentrics=Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",barycentric) &&
            Send<ObjCBool>(Device,"supportsShaderBarycentricCoordinates");
        if (MeshNativeBarycentrics) source.insert(0,"#define EVANS_MESH_NATIVE_BARYCENTRICS 1\n");
        auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new");
        constexpr NSUInteger MeshLanguageVersion=3u<<16; // MTLLanguageVersion3_0.
        Send<void,NSUInteger>(options,"setLanguageVersion:",MeshLanguageVersion); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        Send<void,ObjCBool>(options,"setPreserveInvariance:",ObjCYes);
        ObjCObject error=nullptr;
        MeshLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!MeshLibrary) return Fail("Compile native Metal mesh shaders: "+ErrorDescription(error));
    }
    bool needed[4]{false,frame.NaniteRaster,frame.NaniteRaster,false};
    if (!frame.NaniteRaster) for (const auto& draw:frame.Draws)
        needed[UseMeshTaskStage(frame.MeshDispatch,frame.Scene->Draws[draw.Range].TriangleCount,true)?3:0]=true;
    for (unsigned i=0;i<4;++i) if (needed[i])
    {
        if (!MeshPipelines[i]) MeshPipelines[i]=CreateMeshPipeline(frame.NaniteRaster,frame.NaniteRaster?i-1:0,i==3);
        if (!MeshPipelines[i]) return false;
    }
    return true;
}
void DrawMeshGeometry(ObjCObject encoder,unsigned triangles,bool task)
{
    const auto grid=task?BuildMeshTaskGrid(triangles):BuildMeshShaderGrid(triangles);
    Send<void,MTLSize,MTLSize,MTLSize>(encoder,"drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
        MTLSize{grid.X,grid.Y,grid.Z},MTLSize{task?MeshShaderThreads:1,1,1},MTLSize{MeshShaderThreads,1,1});
}
void DrawNaniteMeshBin(ObjCObject encoder)
{
    // The launch is a conservative bound, while the object shader reads the
    // compact bin count and emits only its actual mesh workgroups.
    const NSUInteger trianglesPerObject=MeshesPerMetalObject*MeshShaderTriangles;
    const NSUInteger groups=(HardwarePlan.Constants.Counts.X+trianglesPerObject-1)/trianglesPerObject;
    Send<void,MTLSize,MTLSize,MTLSize>(encoder,"drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:",
        MTLSize{groups,1,1},MTLSize{1,1,1},MTLSize{MeshShaderThreads,1,1});
}
