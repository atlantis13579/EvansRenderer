// Native instance descriptors use a packed column-major float4x3, followed by
// four uints. This is intentionally independent of the D3D/Vulkan descriptors.
struct LumenMetalInstance
{
    float Transform[12];
    std::uint32_t Options=0,Mask=0,FunctionOffset=0,AccelerationIndex=0;
};
static_assert(sizeof(LumenMetalInstance)==64 && offsetof(LumenMetalInstance,Options)==48,"Metal ray instance ABI changed");
struct LumenMetalAccelerationSizes { NSUInteger Storage=0,Scratch=0,Refit=0; };
struct LumenHardwareRecord
{
    ObjCObject Top=nullptr,Words=nullptr; std::vector<ObjCObject> Bottom;
    ~LumenHardwareRecord() { ReleaseObject(Top); ReleaseObject(Words); for (auto object:Bottom) ReleaseObject(object); }
};
enum class LumenHardwareProgram:unsigned { Distance,Radiance,Probes,Direct,Bounce,Visibility,Count };
ObjCObject LumenHardwareLibrary=nullptr;
std::array<ObjCObject,unsigned(LumenHardwareProgram::Count)> LumenHardwarePipelines{};
void DestroyLumenHardwarePrograms()
{
    for (auto pipeline:LumenHardwarePipelines) ReleaseObject(pipeline); LumenHardwarePipelines={};
    ReleaseObject(LumenHardwareLibrary); LumenHardwareLibrary=nullptr;
}
bool EnsureLumenHardwarePipeline(LumenHardwareProgram program)
{
    auto& pipeline=LumenHardwarePipelines[unsigned(program)]; if (pipeline) return true;
    if (!LumenHardwareLibrary)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","Lumen.metal",source,LastError)) return false;
        const auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new");
        Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
        constexpr NSUInteger LanguageVersion24=(2u<<16)|4u; Send<void,NSUInteger>(options,"setLanguageVersion:",LanguageVersion24);
        const auto key=MakeString("EVANS_LUMEN_HARDWARE_RAYS"),value=Send<ObjCObject,int>(GetClass("NSNumber"),"numberWithInt:",1);
        const auto definitions=Send<ObjCObject,const ObjCObject*,const ObjCObject*,NSUInteger>(GetClass("NSDictionary"),"dictionaryWithObjects:forKeys:count:",&value,&key,1);
        Send<void,ObjCObject>(options,"setPreprocessorMacros:",definitions); ObjCObject error=nullptr;
        LumenHardwareLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error); ReleaseObject(options);
        if (!LumenHardwareLibrary) return Fail("Metal Lumen intersection-query compilation failed: "+ErrorDescription(error));
    }
    const char* names[]{"lumenTraceDistance","lumenTraceRadiance","lumenTraceProbes","lumenLightSurface","lumenBounceSurface","lumenProbeVisibility"};
    const auto function=Send<ObjCObject,ObjCObject>(LumenHardwareLibrary,"newFunctionWithName:",MakeString(names[unsigned(program)]));
    if (!function) return Fail("Metal Lumen hardware kernel is unavailable"); ObjCObject error=nullptr;
    pipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error); ReleaseObject(function);
    if (!pipeline) return Fail("Metal Lumen hardware pipeline failed: "+ErrorDescription(error));
    if (Send<NSUInteger>(pipeline,"maxTotalThreadsPerThreadgroup")<LumenDistanceTraceThreads)
    { ReleaseObject(pipeline); pipeline=nullptr; return Fail("Metal Lumen hardware threadgroup exceeds device limits"); }
    return true;
}
void BindLumenHardware(ObjCObject encoder,const LumenDistanceRecord& distance)
{
    if (!distance.Hardware) return;
    Send<void,ObjCObject,NSUInteger>(encoder,"setAccelerationStructure:atBufferIndex:",distance.Hardware->Top,8);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",distance.Hardware->Words,0,9);
    constexpr NSUInteger ResourceUsageRead=1;
    for (auto bottom:distance.Hardware->Bottom) Send<void,ObjCObject,NSUInteger>(encoder,"useResource:usage:",bottom,ResourceUsageRead);
}
bool SupportsLumenHardwareRayTracing() const
{
    // Intersection queries require Metal 2.4; basic AS support alone also exists
    // on Metal 2.3. Selector availability identifies the newer API generation.
    return Device && Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",Runtime().RegisterSelector("supportsRaytracing")) &&
        Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",Runtime().RegisterSelector("supportsRaytracingFromRender")) &&
        Send<ObjCBool>(Device,"supportsRaytracing");
}
bool LumenAccelerationSizes(ObjCObject descriptor,LumenMetalAccelerationSizes& sizes)
{
#if defined(__x86_64__)
    // objc_msgSend_stret is required for this 24-byte result on Intel macOS.
    // arm64 uses the ordinary message entry point with its indirect-result ABI.
    using Function=void(*)(LumenMetalAccelerationSizes*,ObjCObject,ObjCSelector,ObjCObject);
    const auto function=reinterpret_cast<Function>(dlsym(Runtime().ObjCRuntime,"objc_msgSend_stret"));
    if (!function) return Fail("Metal acceleration-size structure return is unavailable");
    function(&sizes,Device,Runtime().RegisterSelector("accelerationStructureSizesWithDescriptor:"),descriptor);
#else
    sizes=Send<LumenMetalAccelerationSizes,ObjCObject>(Device,"accelerationStructureSizesWithDescriptor:",descriptor);
#endif
    return sizes.Storage!=0 || Fail("Metal acceleration structure reported zero storage");
}
bool BuildLumenHardwareAcceleration(const std::shared_ptr<LumenDistanceRecord>& distance)
{
    const auto& snapshot=distance->Info.Hardware;
    if (!snapshot || snapshot->Instances.empty() || !SupportsLumenHardwareRayTracing()) return true;
    auto record=std::make_shared<LumenHardwareRecord>();
    const auto wordBytes=snapshot->Words.size()*sizeof(unsigned);
    record->Words=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",snapshot->Words.data(),wordBytes,0);
    if (!record->Words) return Fail("Metal Lumen triangle/coverage upload failed");
    struct BuildResources
    {
        ObjCObject Encoder=nullptr; std::vector<ObjCObject> Objects;
        ~BuildResources() { if (Encoder) Send<void>(Encoder,"endEncoding"); for (auto object:Objects) ReleaseObject(object); }
        ObjCObject Keep(ObjCObject object) { if (object) Objects.push_back(object); return object; }
    } work;
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal ray build command buffer is unavailable");
    const auto build=[&](ObjCObject descriptor,ObjCObject& result) {
        LumenMetalAccelerationSizes sizes; if (!LumenAccelerationSizes(descriptor,sizes)) return false;
        result=Send<ObjCObject,NSUInteger>(Device,"newAccelerationStructureWithSize:",sizes.Storage);
        const auto scratch=work.Keep(Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",std::max(NSUInteger{1},sizes.Scratch),StorageModePrivate<<ResourceStorageModeShift));
        if (!result || !scratch) return Fail("Metal Lumen acceleration/scratch allocation failed");
        // Separate encoders establish BLAS-write/TLAS-read dependencies with
        // tracked resources; all input allocations survive command completion.
        work.Encoder=Send<ObjCObject>(command,"accelerationStructureCommandEncoder");
        if (!work.Encoder) return Fail("Metal ray build encoder is unavailable");
        Send<void,ObjCObject,ObjCObject,ObjCObject,NSUInteger>(work.Encoder,"buildAccelerationStructure:descriptor:scratchBuffer:scratchBufferOffset:",result,descriptor,scratch,0);
        Send<void>(work.Encoder,"endEncoding"); work.Encoder=nullptr; return true;
    };
    record->Bottom.resize(snapshot->Meshes.size());
    for (unsigned mesh=0;mesh<snapshot->Meshes.size();++mesh)
    {
        const auto& triangles=snapshot->Meshes[mesh]->Triangles; std::vector<Float3> vertices; vertices.reserve(triangles.size()*3);
        for (const auto& triangle:triangles) for (auto p:{triangle.A,triangle.B,triangle.C}) vertices.push_back({p.X,p.Y,p.Z});
        const auto input=work.Keep(Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",vertices.data(),vertices.size()*sizeof(Float3),0));
        const auto geometry=work.Keep(Send<ObjCObject>(GetClass("MTLAccelerationStructureTriangleGeometryDescriptor"),"new"));
        const auto descriptor=work.Keep(Send<ObjCObject>(GetClass("MTLPrimitiveAccelerationStructureDescriptor"),"new"));
        if (!input || !geometry || !descriptor) return Fail("Metal Lumen triangle build descriptor allocation failed");
        Send<void,ObjCObject>(geometry,"setVertexBuffer:",input); Send<void,NSUInteger>(geometry,"setVertexStride:",sizeof(Float3));
        // Float3 is the original Metal triangle-descriptor default. Keep this
        // compatible with ray-query devices predating explicit vertexFormat.
        Send<void,NSUInteger>(geometry,"setTriangleCount:",triangles.size()); Send<void,ObjCBool>(geometry,"setOpaque:",ObjCNo);
        const auto geometries=Send<ObjCObject,const ObjCObject*,NSUInteger>(GetClass("NSArray"),"arrayWithObjects:count:",&geometry,1);
        Send<void,ObjCObject>(descriptor,"setGeometryDescriptors:",geometries);
        if (!build(descriptor,record->Bottom[mesh])) return false;
    }
    constexpr std::uint32_t DisableTriangleCulling=1,NonOpaque=1u<<3,GeometryMask=1,ShadowMask=2;
    std::vector<LumenMetalInstance> instances(snapshot->Instances.size());
    for (unsigned index=0;index<instances.size();++index)
    {
        const auto& source=snapshot->Instances[index]; auto& result=instances[index];
        for (unsigned column=0;column<4;++column) for (unsigned row=0;row<3;++row) result.Transform[column*3+row]=source.World.Values[row*4+column];
        result.Options=DisableTriangleCulling|NonOpaque; result.Mask=GeometryMask|(source.CastShadow?ShadowMask:0); result.AccelerationIndex=source.Mesh;
    }
    const auto input=work.Keep(Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,"newBufferWithBytes:length:options:",instances.data(),instances.size()*sizeof(instances[0]),0));
    const auto descriptor=work.Keep(Send<ObjCObject>(GetClass("MTLInstanceAccelerationStructureDescriptor"),"new"));
    if (!input || !descriptor) return Fail("Metal Lumen instance descriptor allocation failed");
    const auto structures=Send<ObjCObject,const ObjCObject*,NSUInteger>(GetClass("NSArray"),"arrayWithObjects:count:",record->Bottom.data(),record->Bottom.size());
    Send<void,ObjCObject>(descriptor,"setInstancedAccelerationStructures:",structures); Send<void,ObjCObject>(descriptor,"setInstanceDescriptorBuffer:",input);
    Send<void,NSUInteger>(descriptor,"setInstanceCount:",instances.size()); Send<void,NSUInteger>(descriptor,"setInstanceDescriptorStride:",sizeof(instances[0]));
    if (!build(descriptor,record->Top)) return false;
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal Lumen acceleration build failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    distance->Hardware=std::move(record); return true;
}
