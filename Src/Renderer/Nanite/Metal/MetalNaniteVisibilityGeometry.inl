// Tracked Metal resources publish GPU-assembled attributes between encoders.
// The CPU supplies transforms/ranges only and never maps the world table.
struct NaniteAssemblySource
{
    NaniteVisibilityGeometryConstants Constants; unsigned GroupsX=0,GroupsY=0; BufferHandle Vertex,Index;
};
std::vector<NaniteAssemblySource> NaniteAssemblySources;
ObjCObject NaniteAssemblyLibrary=nullptr,NaniteAssemblyPipeline=nullptr;
void DestroyNaniteVisibilityGeometry()
{ DestroyNaniteInstances();
    NaniteAssemblySources.clear(); ReleaseObject(NaniteAssemblyPipeline); ReleaseObject(NaniteAssemblyLibrary);
    NaniteAssemblyPipeline=NaniteAssemblyLibrary=nullptr;
}
bool PrepareNaniteVisibilityGeometry(const VisibilityScene& scene)
{
    if (!ValidateVisibilityScene(scene,LastError)) return false;
    std::vector<NaniteAssemblySource> sources; sources.reserve(scene.NativeMeshes.size());
    for (unsigned i=0;i<scene.NativeMeshes.size();++i)
    {
        const auto& mesh=scene.NativeMeshes[i]; const auto vertex=Buffers.find(mesh.VertexBuffer.Value),index=Buffers.find(mesh.IndexBuffer.Value);
        if (vertex==Buffers.end() || index==Buffers.end() || !vertex->second.NaniteGeometry || !index->second.NaniteGeometry)
            return Fail("Metal visibility assembly requires native decoded buffers");
        NaniteAssemblySource source; source.Vertex=mesh.VertexBuffer; source.Index=mesh.IndexBuffer;
        if (!BuildNaniteVisibilityGeometryConstants(scene,i,vertex->second.Description,index->second.Description,source.Constants,source.GroupsX,source.GroupsY,LastError)) return false;
        sources.push_back(source);
    }
    if (!NaniteAssemblyLibrary)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","NaniteRaster.metal",source,LastError)) return false;
        auto options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"); Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo); ObjCObject error=nullptr;
        NaniteAssemblyLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!NaniteAssemblyLibrary) return Fail("Compile Metal Nanite visibility assembly: "+ErrorDescription(error));
    }
    if (!NaniteAssemblyPipeline)
    {
        auto function=Send<ObjCObject,ObjCObject>(NaniteAssemblyLibrary,"newFunctionWithName:",MakeString("EvansNaniteVisibilityGeometry")); ObjCObject error=nullptr;
        if (function) NaniteAssemblyPipeline=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function); if (!NaniteAssemblyPipeline) return Fail("Create Metal Nanite visibility assembly: "+ErrorDescription(error));
    }
    if (Send<NSUInteger>(NaniteAssemblyPipeline,"maxTotalThreadsPerThreadgroup")<NaniteRasterThreads)
        return Fail("Metal Nanite visibility assembly exceeds threadgroup limits");
    auto table=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",scene.TriangleCount()*sizeof(VisibilityTriangle),0);
    if (!table) return Fail("Allocate Metal Nanite world table");
    ReleaseObject(VisibilityGeometry); VisibilityGeometry=table; UploadedVisibility.reset(); NaniteAssemblySources=std::move(sources); return true;
}
bool DrawNaniteVisibilityGeometry()
{
    NaniteCopyLocks.reserve(NaniteCopyLocks.size()+NaniteAssemblySources.size()*2);
    for (const auto& source:NaniteAssemblySources) { NaniteCopyLocks.push_back(source.Vertex); NaniteCopyLocks.push_back(source.Index); }
    auto encoder=Send<ObjCObject>(CommandBuffer,"computeCommandEncoder"); if (!encoder) return Fail("Create Metal native visibility assembly encoder");
    Send<void,ObjCObject>(encoder,"setComputePipelineState:",NaniteAssemblyPipeline);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",VisibilityGeometry,0,3);
    for (const auto& source:NaniteAssemblySources)
    {
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&source.Constants,sizeof(source.Constants),0);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",Buffers.at(source.Vertex.Value).Buffer,0,1);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",Buffers.at(source.Index.Value).Buffer,0,2);
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{source.GroupsX,source.GroupsY,1},MTLSize{NaniteRasterThreads,1,1});
        ++PendingFrame.Statistics.NaniteGeometryAssemblyCount;
    }
    Send<void>(encoder,"endEncoding"); return true;
}
