struct LumenCaptureWork : LumenCaptureJob
{
    Impl& Owner; std::shared_ptr<LumenSurfaceRecord> Record; std::shared_ptr<LumenSurfaceWork> Resources;
    VkCommandPool CommandsPool=0; VkCommandBuffer Commands=nullptr; VkFence Completion=0;
    explicit LumenCaptureWork(Impl& owner):Owner(owner) {}
    void ReleaseCommands()
    {
        if (Completion) Owner.DestroyFenceFunction(Owner.Device,Completion,nullptr); Completion=0;
        if (CommandsPool) Owner.DestroyCommandPoolFunction(Owner.Device,CommandsPool,nullptr); CommandsPool=0; Commands=nullptr;
    }
    ~LumenCaptureWork() { ReleaseCommands(); }
};
VkRenderPass LumenCaptureLoadPass=0; PFN_vkGetFenceStatus CaptureGetFenceStatus=nullptr;
LumenCaptureQueue<LumenCaptureWork> LumenCaptureTasks;
bool CollectLumenCapture()
{ return LumenCaptureTasks.Collect([&](auto& work,bool& ready) { return ProbeLumenCapture(work,ready); }); }
bool CancelLumenCaptureUpdate(LumenCaptureTask task)
{ return (LumenCaptureTasks.Cancel(task) || Fail("Invalid Lumen capture task")) && CollectLumenCapture(); }
template<class Handle> bool RetireLumenCaptureInput(Handle handle)
{
    if (!handle || LumenCaptureTasks.Empty()) return true;
    // Source deletion/mutation is exceptional. Drain only submissions borrowing
    // this resource; ordinary frame advances never wait for their capture work.
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    if (!LumenCaptureTasks.Visit([&](auto& work)
    {
        if (!work.Uses(handle)) return true;
        if (FrameOpen) return Fail("Lumen capture inputs cannot change inside a render frame");
        work.Cancelled=true;
        while (work.InFlight)
        {
            bool ready=false; if (!ProbeLumenCapture(work,ready)) return false;
            if (ready) { work.InFlight=false; break; }
            if (std::chrono::steady_clock::now()>=deadline) return Fail("Retiring Lumen capture input timed out");
            std::this_thread::yield();
        }
        return true;
    })) return false;
    return CollectLumenCapture();
}
void DestroyLumenCaptureAsync()
{
    LumenCaptureTasks.Clear();
    if (LumenCaptureLoadPass) DestroyRenderPassFunction(Device,LumenCaptureLoadPass,nullptr); LumenCaptureLoadPass=0;
}
bool ProbeLumenCapture(LumenCaptureWork& work,bool& ready)
{
    const auto status=CaptureGetFenceStatus(Device,work.Completion); ready=status==VK_SUCCESS;
    return status==VK_NOT_READY || Check(status,"Poll capture slice completion");
}
bool EnsureLumenCaptureLoadPass()
{
    // Load/store operations do not affect render-pass compatibility. Preserve
    // earlier cards while sharing the existing capture pipelines/framebuffer.
    // https://docs.vulkan.org/spec/latest/chapters/renderpass.html#renderpass-compatibility
    if (!LumenCaptureLoadPass)
    {
        std::array<VkAttachmentDescription,LumenSurfacePlanes+1> attachments{}; std::array<VkAttachmentReference,LumenSurfacePlanes> colors{};
        for (unsigned slot=0;slot<attachments.size();++slot)
        {
            const bool depth=slot==LumenSurfacePlanes; auto& attachment=attachments[slot];
            attachment.format=depth?VK_FORMAT_D32_SFLOAT:VK_FORMAT_R32G32B32A32_SFLOAT; attachment.samples=VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD; attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout=attachment.finalLayout=depth?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            if (!depth) colors[slot]={slot,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        const VkAttachmentReference depth{LumenSurfacePlanes,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=LumenSurfacePlanes; subpass.pColorAttachments=colors.data(); subpass.pDepthStencilAttachment=&depth;
        VkSubpassDependency dependency{}; dependency.srcSubpass=VK_SUBPASS_EXTERNAL; dependency.dstSubpass=0;
        dependency.srcStageMask=dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstAccessMask=dependency.srcAccessMask|VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; dependency.dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo create{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; create.attachmentCount=unsigned(attachments.size()); create.pAttachments=attachments.data();
        create.subpassCount=1; create.pSubpasses=&subpass; create.dependencyCount=1; create.pDependencies=&dependency;
        if (!Check(CreateRenderPassFunction(Device,&create,nullptr,&LumenCaptureLoadPass),"Create Lumen material render pass")) return false;
    }
    return true;
}
bool BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& output)
{
    if (!LoadDevice(CaptureGetFenceStatus,"vkGetFenceStatus") || !CollectLumenCapture() || !EnsureLumenSurface()) return false;
    auto job=std::make_shared<LumenCaptureWork>(*this);
    if (!job->Prepare(upload,LastError) || !LumenCaptureTasks.CanSubmit(*job,LastError)) return false;
    const auto& layout=job->Layout;
    if (!EnsureLumenCaptureLoadPass()) return false;
    if (layout.Width>DeviceLimits.maxImageDimension2D || layout.Height>DeviceLimits.maxImageDimension2D) return Fail("Capture atlas exceeds Vulkan image dimensions");
    auto work=std::make_shared<LumenSurfaceWork>(*this); job->Resources=work;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        {
            const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid");
            if (std::find(work->MaterialImages.begin(),work->MaterialImages.end(),&found->second.Image)==work->MaterialImages.end()) work->MaterialImages.push_back(&found->second.Image);
        }
    }
    work->Candidate=std::make_shared<LumenSurfaceRecord>(*this); auto& record=*work->Candidate; record.Layout=layout; record.Identity=upload.Identity; record.Captures=upload.Cards;
    std::array<VkImageView,LumenSurfacePlanes+1> views;
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        if (!CreateRawImage(layout.Width,layout.Height,VK_FORMAT_R32G32B32A32_SFLOAT,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,record.Images[plane])) return false;
        views[plane]=record.Images[plane].View;
    }
    if (!CreateRawImage(layout.Width,layout.Height,VK_FORMAT_D32_SFLOAT,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,record.Depth)) return false;
    views[LumenSurfacePlanes]=record.Depth.View;
    if (!CreateFramebuffer(LumenSurfacePass,views.data(),unsigned(views.size()),layout.Width,layout.Height,record.Framebuffer) || !CreateLumenWork(*work,unsigned(job->Constants.size())+1,false)) return false;
    job->Record=work->Candidate;
    unsigned next=0;
    for (const auto& card:upload.Cards) for (unsigned index=card.FirstDraw;index<std::uint64_t(card.FirstDraw)+card.DrawCount;++index)
    {
        const auto& draw=upload.Draws[index]; const auto& constants=job->Constants[next];
        WriteLumenSet(*work,next++,&constants,sizeof(constants),{draw.Diffuse?&Textures.at(draw.Diffuse.Value).Image:nullptr,
            draw.Normal?&Textures.at(draw.Normal.Value).Image:nullptr,draw.Opacity?&Textures.at(draw.Opacity.Value).Image:nullptr},false);
    }
    const ShaderUInt4 extent{layout.Width,layout.Height,LumenSurfaceReadColumns*LumenSurfaceReadThreads,0};
    WriteLumenSet(*work,next,&extent,sizeof(extent),{&record.Images[0],&record.Images[1],&record.Images[2]},true);
    const auto task=LumenCaptureTasks.Insert(job); if (!task) return Fail("Lumen capture task space exhausted"); output=task; return true;
}
bool SubmitLumenCaptureSlice(LumenCaptureWork& job,const LumenCaptureJob::SliceFrame& slice)
{
    const auto& upload=job.Upload; const auto& layout=job.Layout; auto& record=*job.Record; const auto& work=job.Resources;
    job.ReleaseCommands(); VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool.queueFamilyIndex=QueueFamilyIndex;
    if (!Check(CreateCommandPoolFunction(Device,&pool,nullptr,&job.CommandsPool),"Create capture slice command pool")) return false;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; allocate.commandPool=job.CommandsPool; allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount=1;
    if (!Check(AllocateCommandBuffers(Device,&allocate,&job.Commands),"Allocate capture slice commands")) return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (!Check(CreateFenceFunction(Device,&fence,nullptr,&job.Completion),"Create capture slice fence")) return false;
    VkCommandBufferBeginInfo start{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; start.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!Check(BeginCommandBuffer(job.Commands,&start),"Begin capture slice")) return false;
    VkBufferMemoryBarrier uniform{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; uniform.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; uniform.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT;
    uniform.srcQueueFamilyIndex=uniform.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; uniform.buffer=work->Uniform.Buffer; uniform.size=work->Uniform.Size;
    CmdPipelineBarrier(job.Commands,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,1,&uniform,0,nullptr);
    for (auto* image:work->MaterialImages) { auto temporary=*image; TransitionImage(job.Commands,temporary,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); }
    for (const auto& draw:upload.Draws)
    {
        VkBufferMemoryBarrier barriers[2]{}; const BufferHandle handles[]{draw.Vertices,draw.Indices};
        for (unsigned slot=0;slot<2;++slot)
        {
            auto& barrier=barriers[slot]; barrier.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=slot?VK_ACCESS_INDEX_READ_BIT:VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; barrier.buffer=Buffers.at(handles[slot].Value).Allocation.Buffer; barrier.size=Buffers.at(handles[slot].Value).Allocation.Size;
        }
        CmdPipelineBarrier(job.Commands,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,2,barriers,0,nullptr);
    }
    for (auto& image:record.Images) TransitionImage(job.Commands,image,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImage(job.Commands,record.Depth,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    std::array<VkClearValue,LumenSurfacePlanes+1> clear{}; clear[1].color.float32[3]=clear[2].color.float32[3]=1; clear[LumenSurfacePlanes].depthStencil={1,0};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; begin.renderPass=job.Initialized?LumenCaptureLoadPass:LumenSurfacePass; begin.framebuffer=record.Framebuffer;
    begin.renderArea={{0,0},{layout.Width,layout.Height}}; begin.clearValueCount=unsigned(clear.size()); begin.pClearValues=clear.data();
    CmdBeginRenderPass(job.Commands,&begin,VK_SUBPASS_CONTENTS_INLINE); const unsigned stride=layout.CardSide+LumenSurfaceBorder*2;
    unsigned next=job.FirstConstant[slice.Dispatch.X]; const VkDeviceSize offset=0;
    for (unsigned card=slice.Dispatch.X;card<slice.Dispatch.X+slice.Dispatch.Y;++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=(card/layout.Columns)*stride;
        const VkViewport viewport{float(x),float(y),float(stride),float(stride),0,1}; const VkRect2D scissor{{std::int32_t(x),std::int32_t(y)},{stride,stride}};
        CmdSetViewport(job.Commands,0,1,&viewport); CmdSetScissor(job.Commands,0,1,&scissor); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const unsigned raster=(draw.BackFaceCulling?2u:0u)+(RealtimeWorldIsMirrored(draw.World)?1u:0u);
            CmdBindPipeline(job.Commands,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenCapturePipelines[raster]); CmdBindDescriptorSets(job.Commands,VK_PIPELINE_BIND_POINT_GRAPHICS,LumenPipelineLayout,0,1,&work->Sets[next++],0,nullptr);
            CmdBindVertexBuffers(job.Commands,0,1,&Buffers.at(draw.Vertices.Value).Allocation.Buffer,&offset); CmdBindIndexBuffer(job.Commands,Buffers.at(draw.Indices.Value).Allocation.Buffer,0,VK_INDEX_TYPE_UINT32);
            CmdDrawIndexed(job.Commands,draw.IndexCount,1,draw.FirstIndex,0,0);
        }
    }
    CmdEndRenderPass(job.Commands); if (slice.Dispatch.X+slice.Dispatch.Y==layout.CardCount) for (auto& image:record.Images) TransitionImage(job.Commands,image,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!Check(EndCommandBuffer(job.Commands),"End capture slice")) return false;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&job.Commands;
    if (!Check(QueueSubmit(GraphicsQueue,1,&submit,job.Completion),"Submit capture slice")) return false;
    job.InFlight=true; for (auto* image:work->MaterialImages) image->Layout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}
bool AdvanceLumenCaptureUpdate(LumenCaptureTask task,unsigned budget,bool& ready,LumenSurfaceResult& output,LumenCaptureStatistics& statistics)
{
    if (!CollectLumenCapture()) return false;
    return LumenCaptureTasks.Advance(task,budget,ready,output,statistics,LastError,
        [&](auto& work,bool& complete) { return ProbeLumenCapture(work,complete); },
        [&](auto& work,const auto& frame) { return SubmitLumenCaptureSlice(work,frame); },
        [&](auto& work,LumenSurfaceResult& result)
        {
            result.Handle=NewLumenSurfaceHandle(); if (!result.Handle) return Fail("Lumen surface identity space exhausted");
            result.Layout=work.Layout; result.Identity=work.Upload.Identity; result.DrawCalls=work.Statistics.DrawCalls;
            LumenSurfaces.emplace(result.Handle.Value,work.Record); return true;
        });
}
