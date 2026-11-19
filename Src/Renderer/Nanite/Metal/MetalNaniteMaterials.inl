NaniteMaterialWork MaterialPlan;
std::shared_ptr<NaniteFrameBuffer> MaterialMap,MaterialWork;
std::array<ObjCObject,4> MaterialKernels{};
void DestroyNaniteMaterials()
{
    MaterialPlan={}; MaterialMap.reset(); MaterialWork.reset();
    for (auto kernel:MaterialKernels) ReleaseObject(kernel); MaterialKernels={};
}
bool PrepareNaniteMaterials(const VisibilityBufferDesc& frame)
{
    if (!BuildNaniteMaterialWork(frame,CreateInfo.Width,CreateInfo.Height,MaterialPlan,LastError)) return false;
    if (!MaterialPlan.Active()) return true;
    if (!CreateNaniteFrameBuffer(unsigned(MaterialPlan.MaterialBins.size()),MaterialPlan.MaterialBins.data(),false,MaterialMap) ||
        !CreateNaniteFrameBuffer(MaterialPlan.OutputWords,nullptr,true,MaterialWork)) return false;
    if (std::all_of(MaterialKernels.begin(),MaterialKernels.end(),[](auto kernel){return kernel!=nullptr;})) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteMaterials.metal",source,LastError)) return false;
    auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject error=nullptr;
    auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Compile Metal material scheduler: "+ErrorDescription(error));
    constexpr const char* entries[]{"EvansMaterialClear","EvansMaterialClassify","EvansMaterialAllocate","EvansMaterialScatter"};
    for (unsigned pass=0;pass<MaterialKernels.size();++pass) if (!MaterialKernels[pass])
    {
        auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(entries[pass]));
        if (function) MaterialKernels[pass]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!MaterialKernels[pass]) { ReleaseObject(library); return Fail("Create Metal material scheduler: "+ErrorDescription(error)); }
        if (Send<NSUInteger>(MaterialKernels[pass],"maxTotalThreadsPerThreadgroup")<NaniteMaterialThreads)
        { ReleaseObject(MaterialKernels[pass]); MaterialKernels[pass]=nullptr; ReleaseObject(library); return Fail("Metal material scheduler needs a 64-lane workgroup"); }
    }
    ReleaseObject(library); return true;
}
bool ClassifyNaniteMaterials()
{
    if (!MaterialPlan.Active()) return true;
    // Each encoder closes before the next uses its tracked output as input.
    for (unsigned pass=0;pass<MaterialKernels.size();++pass)
    {
        auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal material compute encoder");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",MaterialKernels[pass]);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&MaterialPlan.Constants,sizeof(NaniteMaterialConstants),0);
        const ObjCObject buffers[]{MaterialMap->Buffer,VisibilityGeometry,MaterialWork->Buffer};
        for (NSUInteger index=0;index<3;++index) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",buffers[index],0,index+1);
        Send<void,ObjCObject,NSUInteger>(encoder,"setTexture:atIndex:",VisibilityImages[0],0);
        const auto& c=MaterialPlan.Constants;
        const unsigned x=pass==0?(c.Layout.X+NaniteMaterialThreads-1)/NaniteMaterialThreads:pass==2?1:c.Layout.Y;
        const unsigned y=pass==0 || pass==2?1:c.Layout.Z;
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{x,y,1},MTLSize{NaniteMaterialThreads,1,1});
        Send<void>(encoder,"endEncoding");
    }
    auto& stats=PendingFrame.Statistics; stats.NaniteMaterialDispatchCount+=unsigned(MaterialKernels.size());
    stats.NaniteMaterialBinCount+=unsigned(MaterialPlan.Representatives.size()); stats.VisibilityMaterialDrawCount+=unsigned(MaterialPlan.Representatives.size());
    stats.IndirectDrawCallCount+=unsigned(MaterialPlan.Representatives.size()); return true;
}
