// Shared, default-tracked buffers supply both vertex input and CPU geometry.
// Separate compute encoders order deformation before CSR normal gathering; a
// completed command buffer is required before publishing either representation.
std::array<ObjCObject,2> DisplacementPipelines{};
void DestroyDisplacement()
{ for (auto& pipeline:DisplacementPipelines) { ReleaseObject(pipeline); pipeline=nullptr; } }
bool EnsureDisplacement()
{
    if (DisplacementPipelines[0] && DisplacementPipelines[1]) return true;
    std::string source; if (!LoadTechniqueShaderSource("Metal","DisplacementMapping.metal",source,LastError)) return false;
    ObjCObject options=Send<ObjCObject>(GetClass("MTLCompileOptions"),"new"),error=nullptr;
    Send<void,ObjCBool>(options,"setFastMathEnabled:",ObjCNo);
    ObjCObject library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
    ReleaseObject(options); if (!library) return Fail("Metal displacement compilation failed: "+ErrorDescription(error));
    constexpr const char* entries[]{"Displace","RebuildNormals"};
    for (std::size_t phase=0;phase<DisplacementPipelines.size();++phase) if (!DisplacementPipelines[phase])
    {
        ObjCObject function=Send<ObjCObject,ObjCObject>(library,"newFunctionWithName:",MakeString(entries[phase]));
        if (function) DisplacementPipelines[phase]=Evans::Renderer::CreateCachedMetalComputePipeline(Device,function,&error);
        ReleaseObject(function);
        if (!DisplacementPipelines[phase]) { ReleaseObject(library); return Fail("Metal displacement pipeline failed: "+ErrorDescription(error)); }
        if (Send<NSUInteger>(DisplacementPipelines[phase],"maxTotalThreadsPerThreadgroup")<DisplacementThreadCount)
        { ReleaseObject(library); DestroyDisplacement(); return Fail("Metal displacement group exceeds device capacity"); }
    }
    ReleaseObject(library); return true;
}
bool CreateDisplaced(const DisplacementGeometry& geometry,const Techniques::DisplacementMappingSettings& settings,DisplacedVertexBuffer& output)
{
    DisplacementConstants constants;
    if (!ValidateDisplacementUpload(geometry,settings,constants,LastError) || !EnsureDisplacement()) return false;
    const std::size_t vertexBytes=geometry.Vertices.size()*sizeof(Vertex);
    const void* data[]{geometry.Vertices.data(),geometry.Directions.data(),geometry.Indices.data(),geometry.Neighborhoods.data(),geometry.AdjacentTriangles.data(),nullptr};
    const std::size_t sizes[]{vertexBytes,geometry.Directions.size()*sizeof(Float4),geometry.Indices.size()*sizeof(std::uint32_t),
        geometry.Neighborhoods.size()*sizeof(ShaderUInt4),geometry.AdjacentTriangles.size()*sizeof(std::uint32_t),vertexBytes};
    struct Work
    {
        std::array<ObjCObject,6> Buffers{};
        ~Work() { for (auto buffer:Buffers) ReleaseObject(buffer); }
    } work;
    for (std::size_t slot=0;slot<work.Buffers.size();++slot)
    {
        work.Buffers[slot]=Send<ObjCObject,NSUInteger,NSUInteger>(Device,"newBufferWithLength:options:",sizes[slot],static_cast<NSUInteger>(0));
        if (!work.Buffers[slot]) return Fail("Metal displacement buffer allocation failed");
        void* mapped=Send<void*>(work.Buffers[slot],"contents");
        if (!mapped) return Fail("Metal displacement buffer is not shared/host-visible");
        if (data[slot]) std::memcpy(mapped,data[slot],sizes[slot]);
    }
    BufferResource resource; resource.Description={BufferUsage::Vertex,vertexBytes,false,"Displaced vertices"}; resource.Bytes.resize(vertexBytes);
    DisplacedVertexBuffer candidate; candidate.Vertices.resize(geometry.Vertices.size());
    ObjCObject command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal displacement command buffer creation failed");
    const MTLSize groups{(geometry.Vertices.size()+DisplacementThreadCount-1)/DisplacementThreadCount,1,1},threads{DisplacementThreadCount,1,1};
    for (auto pipeline:DisplacementPipelines)
    {
        ObjCObject encoder=Send<ObjCObject>(command,"computeCommandEncoder");
        if (!encoder) return Fail("Metal displacement encoder creation failed");
        Send<void,ObjCObject>(encoder,"setComputePipelineState:",pipeline);
        for (NSUInteger slot=0;slot<work.Buffers.size();++slot)
            Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setBuffer:offset:atIndex:",work.Buffers[slot],0,slot);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setBytes:length:atIndex:",&constants,sizeof(constants),6);
        Send<void,MTLSize,MTLSize>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",groups,threads); Send<void>(encoder,"endEncoding");
    }
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError)
        return Fail("Metal displacement command failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    const void* mapped=Send<void*>(work.Buffers[5],"contents");
    std::memcpy(candidate.Vertices.data(),mapped,vertexBytes); std::memcpy(resource.Bytes.data(),mapped,vertexBytes);
    if (!ValidateTriangleGeometry(candidate.Vertices,geometry.Indices,LastError)) return false;
    resource.Buffer=work.Buffers[5]; candidate.Buffer={NextBuffer++}; candidate.DispatchCount=static_cast<unsigned>(DisplacementPipelines.size());
    Buffers.emplace(candidate.Buffer.Value,std::move(resource)); work.Buffers[5]=nullptr; output=std::move(candidate); return true;
}
