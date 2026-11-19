// The prepass and pack submission owns its private depth target and SSBO. The
// following demand submission retains that work through its fence, including a
// failed wait, and never round-trips receiver depth through CPU memory.
bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    VirtualShadowRequestConstants request;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,request,LastError) || !EnsureVirtualShadowAtlas()) return false;
    const auto width=upload.Frame.Width,height=upload.Frame.Height;
    const auto groups=(std::uint64_t(width)*height+VirtualShadowGroupWidth-1)/VirtualShadowGroupWidth;
    if (width>DeviceLimits.maxImageDimension2D || height>DeviceLimits.maxImageDimension2D ||
        std::min<std::uint64_t>(groups,VirtualShadowGroupColumns)>DeviceLimits.maxComputeWorkGroupCount[0] ||
        (groups+VirtualShadowGroupColumns-1)/VirtualShadowGroupColumns>DeviceLimits.maxComputeWorkGroupCount[1] ||
        upload.Receivers.size()>=UINT32_MAX/3) return Fail("Receiver depth exceeds Vulkan image, dispatch or descriptor limits");
    auto work=std::make_shared<VirtualShadowAtlasWork>(*this);
    std::vector<VirtualShadowRasterConstants> draws(upload.Receivers.size()+1);
    for (std::size_t draw=0;draw<upload.Receivers.size();++draw)
    {
        const auto& caster=upload.Receivers[draw];
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Receiver buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError) ||
            !BuildVirtualShadowReceiverConstants(upload.Frame,caster,draws[draw],LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        {
            const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Receiver coverage texture is invalid");
            if (std::find(work->CoverageImages.begin(),work->CoverageImages.end(),&found->second.Image)==work->CoverageImages.end()) work->CoverageImages.push_back(&found->second.Image);
        }
    }
    work->Candidate=std::make_shared<VirtualShadowAtlasRecord>(*this); auto& target=*work->Candidate;
    // This target is transient camera depth, never a published residency atlas.
    if (!CreateRawImage(width,height,VK_FORMAT_D32_SFLOAT,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,target.Depth) ||
        !CreateFramebuffer(VirtualAtlasPass,&target.Depth.View,1,width,height,target.Framebuffer) ||
        !CreateVirtualAtlasWork(*work,static_cast<unsigned>(draws.size()),std::size_t(width)*height)) return false;
    const auto readSet=static_cast<unsigned>(upload.Receivers.size()); draws[readSet].Extent={width,height,VirtualShadowGroupColumns*VirtualShadowGroupWidth,0};
    for (unsigned draw=0;draw<readSet;++draw)
    {
        const auto& caster=upload.Receivers[draw];
        WriteVirtualAtlasDraw(*work,draw,draws[draw],caster.Diffuse?&Textures.at(caster.Diffuse.Value).Image:nullptr,
            caster.Opacity?&Textures.at(caster.Opacity.Value).Image:nullptr,nullptr);
    }
    WriteVirtualAtlasDraw(*work,readSet,draws[readSet],nullptr,nullptr,&target.Depth);
    if (!BeginImmediate()) return false; VirtualAtlasUniformBarrier(*work);
    for (auto* image:work->CoverageImages) { auto view=*image; TransitionImage(CommandBuffer,view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); }
    for (const auto& caster:upload.Receivers)
    {
        VkBufferMemoryBarrier visible[2]{}; const BufferHandle buffers[]{caster.Vertices,caster.Indices};
        for (unsigned slot=0;slot<2;++slot)
        {
            auto& barrier=visible[slot]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask=slot?VK_ACCESS_INDEX_READ_BIT:VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer=Buffers.at(buffers[slot].Value).Allocation.Buffer; barrier.size=Buffers.at(buffers[slot].Value).Allocation.Size;
        }
        CmdPipelineBarrier(CommandBuffer,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,2,visible,0,nullptr);
    }
    TransitionImage(CommandBuffer,target.Depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    VkClearValue clear{}; clear.depthStencil={1,0}; VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass=VirtualAtlasPass; begin.framebuffer=target.Framebuffer; begin.renderArea={{0,0},{width,height}}; begin.clearValueCount=1; begin.pClearValues=&clear;
    CmdBeginRenderPass(CommandBuffer,&begin,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,float(width),float(height),0,1}; const VkRect2D scissor{{0,0},{width,height}}; const VkDeviceSize offset=0;
    CmdSetViewport(CommandBuffer,0,1,&viewport); CmdSetScissor(CommandBuffer,0,1,&scissor);
    for (unsigned draw=0;draw<readSet;++draw)
    {
        const auto& caster=upload.Receivers[draw]; const unsigned raster=caster.BackFaceCulling?(RealtimeWorldIsMirrored(caster.World)?2:1):0;
        CmdBindPipeline(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VirtualAtlasRaster[raster]);
        CmdBindDescriptorSets(CommandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,VirtualAtlasLayout,0,1,&work->Sets[draw],0,nullptr);
        CmdBindVertexBuffers(CommandBuffer,0,1,&Buffers.at(caster.Vertices.Value).Allocation.Buffer,&offset);
        CmdBindIndexBuffer(CommandBuffer,Buffers.at(caster.Indices.Value).Allocation.Buffer,0,VK_INDEX_TYPE_UINT32); CmdDrawIndexed(CommandBuffer,caster.IndexCount,1,caster.FirstIndex,0,0);
    }
    CmdEndRenderPass(CommandBuffer); TransitionImage(CommandBuffer,target.Depth,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!EncodeVirtualDepthPack(*work,width,height,readSet) || !CompleteVirtualAtlasWork(work,nullptr)) return false;
    return BuildVirtualShadowRequests({upload.Frame,{}},output,&work->Feedback,work);
}
