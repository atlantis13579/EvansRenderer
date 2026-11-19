void*(EVANS_GL_CALL* CaptureFenceSync)(GLenum,GLbitfield)=nullptr;
GLenum(EVANS_GL_CALL* CaptureClientWaitSync)(void*,GLbitfield,std::uint64_t)=nullptr;
void(EVANS_GL_CALL* CaptureDeleteSync)(void*)=nullptr;
void(EVANS_GL_CALL* CaptureFlush)()=nullptr;
static constexpr GLenum CaptureSyncComplete=0x9117,CaptureAlreadySignaled=0x911A,CaptureTimeoutExpired=0x911B,CaptureConditionSatisfied=0x911C;
struct LumenCaptureWork : LumenCaptureJob
{
    Impl& Owner; std::shared_ptr<LumenSurfaceRecord> Record; std::shared_ptr<LumenCaptureBuffers> Resources;
    void* Completion=nullptr; bool Unfenced=false;
    explicit LumenCaptureWork(Impl& owner):Owner(owner) {}
    ~LumenCaptureWork() { if (Completion) Owner.CaptureDeleteSync(Completion); }
};
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
void DestroyLumenCaptureAsync() { LumenCaptureTasks.Clear(); }
bool ProbeLumenCapture(LumenCaptureWork& work,bool& ready)
{
    if (work.Unfenced) return Fail("Lumen capture has no ES completion fence");
    const auto status=CaptureClientWaitSync(work.Completion,0,0);
    ready=status==CaptureAlreadySignaled || status==CaptureConditionSatisfied;
    return ready || status==CaptureTimeoutExpired || Fail("ES capture slice fence query failed");
}
bool BeginLumenCaptureUpdate(const LumenSurfaceUpload& upload,LumenCaptureTask& output)
{
    if (!LoadGL(CaptureFenceSync,"glFenceSync") || !LoadGL(CaptureClientWaitSync,"glClientWaitSync") ||
        !LoadGL(CaptureDeleteSync,"glDeleteSync") || !LoadGL(CaptureFlush,"glFlush") || !CollectLumenCapture() || !EnsureLumenSurface()) return false;
    auto job=std::make_shared<LumenCaptureWork>(*this);
    if (!job->Prepare(upload,LastError) || !LumenCaptureTasks.CanSubmit(*job,LastError)) return false;
    const auto& layout=job->Layout;
    for (const auto& draw:upload.Draws)
    {
        const auto vertices=Buffers.find(draw.Vertices.Value),indices=Buffers.find(draw.Indices.Value);
        if (vertices==Buffers.end() || indices==Buffers.end()) return Fail("Lumen material draw has invalid buffers");
        if (!ValidateLumenSurfaceDraw(draw,vertices->second.Description,indices->second.Description,indices->second.Bytes,LastError)) return false;
        for (auto texture:{draw.Diffuse,draw.Normal,draw.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Lumen material texture is invalid"); }
    }
    LumenCaptureState state(*this);
    auto record=std::make_shared<LumenSurfaceRecord>(*this); record->Layout=layout; record->Identity=upload.Identity; record->Captures=upload.Cards;
    GlBindBuffer(TSRPixelUnpackBuffer,0); GlActiveTexture(GL_TEXTURE0); LumenBindSampler(0,0);
    GlGenFramebuffers(1,&record->Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,record->Framebuffer);
    GlGenTextures(LumenSurfacePlanes,record->Images.data()); GLenum targets[LumenSurfacePlanes];
    for (unsigned plane=0;plane<LumenSurfacePlanes;++plane)
    {
        GlBindTexture(GL_TEXTURE_2D,record->Images[plane]); GlTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,layout.Width,layout.Height,0,GL_RGBA,GL_FLOAT,nullptr);
        GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        targets[plane]=GL_COLOR_ATTACHMENT0+plane; GlFramebufferTexture2D(GL_FRAMEBUFFER,targets[plane],GL_TEXTURE_2D,record->Images[plane],0);
    }
    GlGenTextures(1,&record->Depth); GlBindTexture(GL_TEXTURE_2D,record->Depth);
    GlTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,layout.Width,layout.Height,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,record->Depth,0); GlDrawBuffers(LumenSurfacePlanes,targets);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create Lumen material atlas")) return Fail("OpenGL ES requires three renderable FP32 Lumen material targets");
    job->Record=record; job->Resources=std::make_shared<LumenCaptureBuffers>(*this); auto& buffers=*job->Resources;
    GlGenVertexArrays(1,&buffers.Array); GlBindVertexArray(buffers.Array); GlGenBuffers(1,&buffers.Buffers[0]);
    GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Buffers[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(LumenSurfaceConstants),nullptr,GL_DYNAMIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Buffers[0]);
    if (!CheckGL("Allocate capture slice resources")) return false;
    const auto task=LumenCaptureTasks.Insert(job); if (!task) return Fail("Lumen capture task space exhausted"); output=task; return true;
}
bool SubmitLumenCaptureSlice(LumenCaptureWork& work,const LumenCaptureJob::SliceFrame& slice)
{
    if (work.Completion) { CaptureDeleteSync(work.Completion); work.Completion=nullptr; }
    const auto& upload=work.Upload; const auto& layout=work.Layout; const auto& record=work.Record;
    LumenCaptureState state(*this); GlBindFramebuffer(GL_FRAMEBUFFER,record->Framebuffer);
    work.InFlight=true;
    for (auto flag:{LumenScissorTest,GL_BLEND,GL_DITHER,GL_POLYGON_OFFSET_FILL,GL_SAMPLE_MASK,LumenStencilTest}) GlDisable(flag);
    GlEnable(GL_DEPTH_TEST); GlDepthFunc(GL_LEQUAL); GlDepthMask(GL_TRUE_VALUE); GlColorMask(GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE,GL_TRUE_VALUE); LumenDepthRange(0,1);
    const float clear[LumenSurfacePlanes][4]{{0,0,0,0},{0,0,0,1},{0,0,0,1}};
    if (!work.Initialized) for (unsigned plane=0;plane<LumenSurfacePlanes;++plane) GlClearBufferfv(GL_COLOR,GLint(plane),clear[plane]);
    if (!work.Initialized) { GlClearDepthf(1); GlClear(GL_DEPTH_BUFFER_BIT); } GlEnable(LumenScissorTest); GlCullFace(GL_BACK);
    GlBindVertexArray(work.Resources->Array); GlBindBuffer(GL_UNIFORM_BUFFER,work.Resources->Buffers[0]); GlBindBufferBase(GL_UNIFORM_BUFFER,0,work.Resources->Buffers[0]);
    GlUseProgram(LumenSurfacePrograms[0]); const char* names[]{"DiffuseImage","NormalImage","OpacityImage"};
    for (unsigned slot=0;slot<LumenSurfacePlanes;++slot) { GlUniform1i(GlGetUniformLocation(LumenSurfacePrograms[0],names[slot]),GLint(slot)); LumenBindSampler(slot,0); }
    for (GLuint attribute=0;attribute<4;++attribute) GlEnableVertexAttribArray(attribute);
    const unsigned stride=layout.CardSide+LumenSurfaceBorder*2;
    for (unsigned card=slice.Dispatch.X;card<slice.Dispatch.X+slice.Dispatch.Y;++card)
    {
        const unsigned x=(card%layout.Columns)*stride,y=layout.Height-(card/layout.Columns+1)*stride;
        GlViewport(x,y,stride,stride); LumenScissor(x,y,stride,stride); const auto& capture=upload.Cards[card];
        for (unsigned index=capture.FirstDraw;index<std::uint64_t(capture.FirstDraw)+capture.DrawCount;++index)
        {
            const auto& draw=upload.Draws[index]; const auto& constants=work.Constants[work.FirstConstant[card]+index-capture.FirstDraw];
            GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(constants),&constants); (draw.BackFaceCulling?GlEnable:GlDisable)(GL_CULL_FACE); GlFrontFace(RealtimeWorldIsMirrored(draw.World)?GL_CW:GL_CCW);
            GlBindBuffer(GL_ARRAY_BUFFER,Buffers.at(draw.Vertices.Value).Buffer); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,Buffers.at(draw.Indices.Value).Buffer);
            GlVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Position)));
            GlVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Normal)));
            GlVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Color)));
            GlVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,TexCoord)));
            const TextureHandle textures[]{draw.Diffuse,draw.Normal,draw.Opacity};
            for (unsigned slot=0;slot<LumenSurfacePlanes;++slot) { GlActiveTexture(GL_TEXTURE0+slot); GlBindTexture(GL_TEXTURE_2D,textures[slot]?Textures.at(textures[slot].Value).Texture:0); }
            GlDrawElements(GL_TRIANGLES,draw.IndexCount,GL_UNSIGNED_INT,reinterpret_cast<const void*>(std::size_t(draw.FirstIndex)*sizeof(std::uint32_t)));
        }
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,0);
    work.Completion=CaptureFenceSync(CaptureSyncComplete,0); CaptureFlush();
    if (!work.Completion) { work.Unfenced=true; return Fail("Create capture slice fence failed"); }
    return CheckGL("Submit Lumen capture slice");
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
