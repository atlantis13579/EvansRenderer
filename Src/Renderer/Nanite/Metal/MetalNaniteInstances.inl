NaniteInstanceWork InstancePlan;
std::vector<std::shared_ptr<NaniteFrameBuffer>> InstanceBuffers;
std::array<ObjCObject,3> InstanceKernels{};
void DestroyNaniteInstances()
{
    InstancePlan={}; InstanceBuffers.clear();
    for (auto kernel:InstanceKernels) ReleaseObject(kernel); InstanceKernels={};
}
bool PrepareNaniteInstances(const VisibilityBufferDesc& frame)
{
    std::vector<NaniteVisibilityGeometryConstants> constants;
    if (frame.GpuInstances) for (const auto& source:NaniteAssemblySources) constants.push_back(source.Constants);
    if (!BuildNaniteInstanceWork(frame,constants,InstancePlan,LastError)) return false;
    InstanceBuffers.clear(); if (InstancePlan.Batches.empty()) return true;
    for (const auto& batch:InstancePlan.Batches)
    {
        std::shared_ptr<NaniteFrameBuffer> work;
        if (!CreateNaniteFrameBuffer(unsigned(batch.Upload.size()),batch.Upload.data(),true,work)) return false;
        InstanceBuffers.push_back(std::move(work));
    }
    if (std::all_of(InstanceKernels.begin(),InstanceKernels.end(),[](auto kernel){return kernel!=nullptr;})) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteInstances.metal",source,LastError)) return false;
    auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo); ObjCObject error=nullptr;
    auto library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Compile Metal instance schedule: "+ErrorDescription(error));
    constexpr const char* entries[]{"EvansInstanceCount","EvansInstanceScan","EvansInstanceAssemble"};
    for (unsigned pass=0;pass<InstanceKernels.size();++pass) if (!InstanceKernels[pass])
    {
        auto function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(entries[pass]));
        if (function) InstanceKernels[pass]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!InstanceKernels[pass]) { ReleaseObject(library); return Fail("Create Metal instance schedule: "+ErrorDescription(error)); }
        if (Send<NSUInteger>(InstanceKernels[pass],"maxTotalThreadsPerThreadgroup")<(pass==1?1:NaniteRasterThreads))
        { ReleaseObject(InstanceKernels[pass]); InstanceKernels[pass]=nullptr; ReleaseObject(library); return Fail("Metal instance schedule exceeds workgroup limits"); }
    }
    ReleaseObject(library); return true;
}
bool DrawNaniteInstances()
{
    for (unsigned family=0;family<InstancePlan.Batches.size();++family)
    {
        const auto& batch=InstancePlan.Batches[family]; const auto work=InstanceBuffers[family]->Buffer;
        NaniteCopyLocks.push_back(batch.Vertex); NaniteCopyLocks.push_back(batch.Index);
        // Separate tracked encoders publish counts, then arguments, then geometry.
        for (unsigned pass=0;pass<InstanceKernels.size();++pass)
        {
            auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal instance schedule encoder");
            Send<void,ObjCObject>(encoder,"setComputePipelineState:",InstanceKernels[pass]);
            Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&batch.Parameters,sizeof(batch.Parameters),0);
            const ObjCObject inputs[]{pass==2?Buffers.at(batch.Vertex.Value).Buffer:NaniteRasterDecisions->Buffer,Buffers.at(batch.Index.Value).Buffer,VisibilityGeometry,work};
            for (NSUInteger binding=0;binding<4;++binding) Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",inputs[binding],0,binding+1);
            if (pass==2)
                Send<void,ObjCObject,NSUInteger,MTLSize>(encoder,"dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",work,0,MTLSize{NaniteRasterThreads,1,1});
            else
                Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{pass?1:(batch.Parameters.X+NaniteRasterThreads-1)/NaniteRasterThreads,1,1},MTLSize{pass?1:NaniteRasterThreads,1,1});
            Send<void>(encoder,"endEncoding");
        }
        ++PendingFrame.Statistics.NaniteGeometryAssemblyCount; PendingFrame.Statistics.NaniteInstanceDispatchCount+=unsigned(InstanceKernels.size());
    }
    return true;
}
