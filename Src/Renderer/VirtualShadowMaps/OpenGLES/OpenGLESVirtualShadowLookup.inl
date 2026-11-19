// The frame retains the immutable atlas until subsequent native work can safely
// replace its bindings. Releasing a public handle cannot invalidate encoded draws.
std::shared_ptr<VirtualShadowAtlasRecord> FrameVirtualShadowAtlas;
VirtualShadowSamplingFrame FrameVirtualShadowSampling;
bool PrepareVirtualShadowSampling(const VirtualShadowSamplingFrame& frame,const CascadedShadowState& cascades)
{
    if (!ValidateVirtualShadowSamplingFrame(frame,LastError)) return false;
    std::shared_ptr<VirtualShadowAtlasRecord> candidate;
    if (frame.Atlas)
    {
        if (cascades.Frame.CascadeCount) return Fail("Virtual and cascaded shadow representations are mutually exclusive");
        if (FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::Hard && FrameShadowFilter.Mode!=Techniques::ShadowFilterMode::PCF)
            return Fail("Virtual scene sampling currently requires hard or PCF filtering");
        const auto found=VirtualShadowAtlases.find(frame.Atlas.Value);
        if (found==VirtualShadowAtlases.end()) return Fail("Virtual scene sampling requires a current atlas handle");
        candidate=found->second;
    }
    FrameVirtualShadowAtlas=std::move(candidate); FrameVirtualShadowSampling=frame; return true;
}
bool CreateVirtualShadowDirectory(VirtualShadowAtlasRecord& record)
{
    VirtualShadowDirectory directory; if (!BuildVirtualShadowDirectory(record.Info,directory,LastError)) return false;
    VirtualAtlasState state(*this); GlBindBuffer(TSRPixelUnpackBuffer,0); GlPixelStorei(AtlasUnpackAlignment,sizeof(float)); GlPixelStorei(AtlasUnpackRowLength,0);
    GlActiveTexture(GL_TEXTURE0); VirtualAtlasBindSampler(0,0); GlGenTextures(1,&record.Directory); GlBindTexture(GL_TEXTURE_2D,record.Directory);
    GlTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,directory.Width,directory.Height,0,GL_DEPTH_COMPONENT,GL_FLOAT,directory.Depth.data());
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,AtlasCompareMode,GL_NONE_VALUE);
    return CheckGL("Upload immutable virtual page directory");
}
bool SampleVirtualShadowAtlas(VirtualShadowAtlasHandle handle,const VirtualShadowLookupUpload& upload,VirtualShadowLookupResult& output)
{
    const auto found=VirtualShadowAtlases.find(handle.Value); if (found==VirtualShadowAtlases.end()) return Fail("Virtual lookup atlas handle is invalid");
    const auto record=found->second; VirtualShadowLookupConstants constants;
    if (!BuildVirtualShadowLookupConstants(record->Info,upload,constants,LastError) || !EnsureVirtualShadowAtlas()) return false;
    const auto bytes=upload.Samples.size()*sizeof(Float4); const auto groups=(constants.Filtering.Z+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
    GLint limit=0,blocks=0,groupsX=0,groupsY=0; GlGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&limit); GlGetIntegerv(VirtualComputeBlocks,&blocks);
    GpuGetIntegerIndexed(VirtualGroupCount,0,&groupsX); GpuGetIntegerIndexed(VirtualGroupCount,1,&groupsY);
    if (limit<=0 || bytes>std::size_t(limit) || blocks<2 || groupsX<=0 || groupsY<=0 ||
        std::min(groups,VirtualShadowGroupColumns)>unsigned(groupsX) || (groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns>unsigned(groupsY)) return Fail("ES virtual lookup exceeds compute storage or dispatch limits");
    VirtualAtlasState state(*this); VirtualAtlasBuffers inputs(*this),feedback(*this);
    GlGenBuffers(2,inputs.Buffers.data()); GlGenBuffers(1,feedback.Buffers.data());
    GlBindBuffer(GL_UNIFORM_BUFFER,inputs.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(constants),&constants,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,inputs.Buffers[0]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,inputs.Buffers[1]); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(bytes),upload.Samples.data(),GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,inputs.Buffers[1]);
    GlBindBuffer(GL_SHADER_STORAGE_BUFFER,feedback.Buffers[0]); GlBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(bytes),nullptr,GL_STATIC_DRAW); GlBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,feedback.Buffers[0]);
    GlUseProgram(VirtualAtlasPrograms[3]); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[3],"PhysicalDepth"),0); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[3],"PageDirectory"),1);
    GlActiveTexture(GL_TEXTURE0); VirtualAtlasBindSampler(0,0); GlBindTexture(GL_TEXTURE_2D,record->Depth);
    GlActiveTexture(GL_TEXTURE0+1); VirtualAtlasBindSampler(1,0); GlBindTexture(GL_TEXTURE_2D,record->Directory);
    if (!CheckGL("Prepare virtual lookup")) return false;
    GpuDispatch(std::min(groups,VirtualShadowGroupColumns),(groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns,1); GpuMemoryBarrier(GpuStorageBarrier|VirtualBufferBarrier);
    const auto* mapped=VirtualMapBuffer(GL_SHADER_STORAGE_BUFFER,0,static_cast<GLsizeiptr>(bytes),VirtualMapRead); if (!mapped) return Fail("ES virtual lookup feedback is not readable");
    VirtualShadowLookupResult result; result.Samples.resize(upload.Samples.size()); std::memcpy(result.Samples.data(),mapped,bytes);
    if (!VirtualUnmapBuffer(GL_SHADER_STORAGE_BUFFER) || !CheckGL("Read virtual lookup")) return Fail("ES virtual lookup feedback was invalidated");
    if (!ValidateVirtualShadowLookupResult(record->Info,upload,result,LastError)) return false; output=std::move(result); return true;
}
