// Shared, hazard-tracked buffers retain the GPU-produced vertex and index
// streams. Separate encoders order shared edge writes before evaluation, and
// command completion precedes publication of both host mirrors.
std::array<ObjCObject,3> TessellationPipelines{};
void DestroyTessellation()
{ for (auto& pipeline:TessellationPipelines) { ReleaseObject(pipeline); pipeline=nullptr; } }
bool EnsureTessellation()
{
    if (TessellationPipelines[0] && TessellationPipelines[1] && TessellationPipelines[2]) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","Tessellation.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr;
    Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Metal PN compilation failed: "+ErrorDescription(error));
    constexpr const char* entries[]{"BuildEdges","BuildVertices","BuildIndices"};
    for (std::size_t phase=0;phase<TessellationPipelines.size();++phase) if (!TessellationPipelines[phase])
    {
        ObjCObject function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(entries[phase]));
        if (function) TessellationPipelines[phase]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!TessellationPipelines[phase]) { ReleaseObject(library); return Fail("Metal PN pipeline failed: "+ErrorDescription(error)); }
        if (Send<NSUInteger>(TessellationPipelines[phase],"maxTotalThreadsPerThreadgroup")<TessellationThreadCount)
        { ReleaseObject(library); DestroyTessellation(); return Fail("Metal PN group exceeds device capacity"); }
    }
    ReleaseObject(library); return true;
}
bool CreateTessellated(const PNTessellationGeometry& geometry,const Techniques::TessellationSettings& settings,TessellatedGeometryBuffers& output)
{
    TessellationConstants constants;
    if (!ValidateTessellationUpload(geometry,settings,constants,LastError) || !EnsureTessellation()) return false;
    const std::size_t vertexBytes=std::size_t(constants.Outputs.X)*sizeof(Vertex),indexBytes=std::size_t(constants.Outputs.Y)*sizeof(std::uint32_t);
    const void* data[]{geometry.Vertices.data(),geometry.Patches.data(),geometry.PatchEdges.data(),geometry.AttributeEdges.data(),geometry.GeometricEdges.data(),nullptr,nullptr,nullptr};
    const std::size_t sizes[]{geometry.Vertices.size()*sizeof(Vertex),geometry.Patches.size()*sizeof(ShaderUInt4),geometry.PatchEdges.size()*sizeof(ShaderUInt4),
        geometry.AttributeEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*sizeof(ShaderUInt4),geometry.GeometricEdges.size()*2*sizeof(Float4),vertexBytes,indexBytes};
    struct Work
    {
        std::array<ObjCObject,8> Buffers{};
        ~Work() { for (auto buffer:Buffers) ReleaseObject(buffer); }
    } work;
    for (std::size_t slot=0;slot<work.Buffers.size();++slot)
    {
        work.Buffers[slot]=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",sizes[slot],static_cast<NSUInteger>(0));
        if (!work.Buffers[slot]) return Fail("Metal PN buffer allocation failed");
        void* mapped=Send<void*>(work.Buffers[slot],"contents"); if (!mapped) return Fail("Metal PN allocation is not shared/host-visible");
        if (data[slot]) std::memcpy(mapped,data[slot],sizes[slot]);
    }
    ObjCObject command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal PN command buffer creation failed");
    const unsigned counts[]{constants.Inputs.W,constants.Outputs.X,constants.Outputs.Y/3};
    const MTLSize threads{TessellationThreadCount,1,1};
    for (std::size_t phase=0;phase<TessellationPipelines.size();++phase)
    {
        ObjCObject encoder=Send<ObjCObject>(command,"computeCommandEncoder"); if (!encoder) return Fail("Metal PN encoder creation failed");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",TessellationPipelines[phase]);
        for (NSUInteger slot=0;slot<work.Buffers.size();++slot)
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Buffers[slot],0,slot);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),8);
        const MTLSize groups{(counts[phase]+TessellationThreadCount-1)/TessellationThreadCount,1,1};
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",groups,threads); Send<void>(encoder,"endEncoding");
    }
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError)
        return Fail("Metal PN command failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    TessellatedGeometryBuffers candidate; candidate.Vertices.resize(constants.Outputs.X); candidate.Indices.resize(constants.Outputs.Y);
    std::array<BufferResource,2> resources;
    resources[0].Description={BufferUsage::Vertex,vertexBytes,false,"PN tessellated vertices"}; resources[1].Description={BufferUsage::Index,indexBytes,false,"PN tessellated indices"};
    void* mirrors[]{candidate.Vertices.data(),candidate.Indices.data()};
    for (std::size_t stream=0;stream<resources.size();++stream)
    {
        auto& resource=resources[stream]; const auto bytes=resource.Description.SizeInBytes; resource.Bytes.resize(bytes);
        const void* mapped=Send<void*>(work.Buffers[6+stream],"contents");
        std::memcpy(mirrors[stream],mapped,bytes); std::memcpy(resource.Bytes.data(),mapped,bytes); resource.Buffer=work.Buffers[6+stream];
    }
    if (!ValidateTriangleGeometry(candidate.Vertices,candidate.Indices,LastError)) return false;
    candidate.VertexBuffer={NextBuffer++}; candidate.IndexBuffer={NextBuffer++}; candidate.DispatchCount=3;
    try { Buffers.emplace(candidate.VertexBuffer.Value,std::move(resources[0])); Buffers.emplace(candidate.IndexBuffer.Value,std::move(resources[1])); }
    catch (...) { Buffers.erase(candidate.VertexBuffer.Value); throw; }
    work.Buffers[6]=work.Buffers[7]=nullptr; output=std::move(candidate); return true;
}
