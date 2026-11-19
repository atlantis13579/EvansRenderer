// The depth target follows GL's bottom-left framebuffer convention. The native
// packing stage converts it once to canonical top-left receiver indexing.
bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    VirtualShadowRequestConstants request;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,request,LastError) || !EnsureVirtualShadowAtlas()) return false;
    const auto width=upload.Frame.Width,height=upload.Frame.Height; GLint limit=0; GlGetIntegerv(AtlasMaximumTexture,&limit);
    if (limit<=0 || width>unsigned(limit) || height>unsigned(limit)) return Fail("Receiver depth exceeds ES image dimensions");
    std::vector<VirtualShadowRasterConstants> draws(upload.Receivers.size());
    for (std::size_t draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw];
        const auto vertex=Buffers.find(caster.Vertices.Value),index=Buffers.find(caster.Indices.Value);
        if (vertex==Buffers.end() || index==Buffers.end()) return Fail("Receiver buffers are invalid");
        if (!ValidateVirtualShadowCaster(caster,vertex->second.Description,index->second.Description,index->second.Bytes,vertex->second.NaniteGeometry && index->second.NaniteGeometry,LastError) ||
            !BuildVirtualShadowReceiverConstants(upload.Frame,caster,draws[draw],LastError)) return false;
        for (auto texture:{caster.Diffuse,caster.Opacity}) if (texture)
        { const auto found=Textures.find(texture.Value); if (found==Textures.end() || found->second.Description.Format!=TextureFormat::RGBA8) return Fail("Receiver coverage texture is invalid"); }
    }
    VirtualAtlasState state(*this); VirtualAtlasBuffers buffers(*this),packed(*this); VirtualShadowAtlasRecord target(*this);
    GlBindBuffer(TSRPixelUnpackBuffer,0); GlActiveTexture(GL_TEXTURE0+2); VirtualAtlasBindSampler(2,0);
    GlGenTextures(1,&target.Depth); GlBindTexture(GL_TEXTURE_2D,target.Depth);
    GlTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,width,height,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE); GlTexParameteri(GL_TEXTURE_2D,AtlasCompareMode,GL_NONE_VALUE);
    GlGenFramebuffers(1,&target.Framebuffer); GlBindFramebuffer(GL_FRAMEBUFFER,target.Framebuffer);
    GlFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,target.Depth,0);
    const GLenum none=GL_NONE_VALUE; GlDrawBuffers(1,&none); GlReadBuffer(GL_NONE_VALUE);
    if (GlCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE || !CheckGL("Create receiver depth target")) return Fail("ES receiver depth framebuffer is incomplete");
    GlBindTexture(GL_TEXTURE_2D,0); GlDisable(AtlasScissorTest); GlDisable(GL_BLEND); GlDisable(GL_DITHER); GlDisable(GL_POLYGON_OFFSET_FILL); GlDisable(GL_SAMPLE_MASK);
    GlEnable(GL_DEPTH_TEST); GlDepthMask(GL_TRUE_VALUE); GlDepthFunc(GL_LEQUAL); VirtualAtlasDepthRange(0,1); GlClearDepthf(1); GlClear(GL_DEPTH_BUFFER_BIT);
    GlColorMask(GL_FALSE_VALUE,GL_FALSE_VALUE,GL_FALSE_VALUE,GL_FALSE_VALUE); GlViewport(0,0,width,height); GlCullFace(GL_BACK);
    GlGenVertexArrays(1,&buffers.Array); GlBindVertexArray(buffers.Array); for (GLuint attribute:{0U,2U,3U}) GlEnableVertexAttribArray(attribute);
    GlGenBuffers(1,&buffers.Buffers[0]); GlBindBuffer(GL_UNIFORM_BUFFER,buffers.Buffers[0]);
    GlBufferData(GL_UNIFORM_BUFFER,sizeof(VirtualShadowRasterConstants),nullptr,GL_STATIC_DRAW); GlBindBufferBase(GL_UNIFORM_BUFFER,0,buffers.Buffers[0]);
    GlUseProgram(VirtualAtlasPrograms[0]); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[0],"DiffuseImage"),0); GlUniform1i(GlGetUniformLocation(VirtualAtlasPrograms[0],"OpacityImage"),1);
    VirtualAtlasBindSampler(0,0); VirtualAtlasBindSampler(1,0);
    for (std::size_t draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw]; GlBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(draws[draw]),&draws[draw]);
        (caster.BackFaceCulling?GlEnable:GlDisable)(GL_CULL_FACE); GlFrontFace(RealtimeWorldIsMirrored(caster.World)?GL_CW:GL_CCW);
        GlBindBuffer(GL_ARRAY_BUFFER,Buffers.at(caster.Vertices.Value).Buffer); GlBindBuffer(GL_ELEMENT_ARRAY_BUFFER,Buffers.at(caster.Indices.Value).Buffer);
        GlVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Position)));
        GlVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,Color)));
        GlVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE_VALUE,sizeof(Vertex),reinterpret_cast<const void*>(offsetof(Vertex,TexCoord)));
        GlActiveTexture(GL_TEXTURE0); GlBindTexture(GL_TEXTURE_2D,caster.Diffuse?Textures.at(caster.Diffuse.Value).Texture:0);
        GlActiveTexture(GL_TEXTURE0+1); GlBindTexture(GL_TEXTURE_2D,caster.Opacity?Textures.at(caster.Opacity.Value).Texture:0);
        GlDrawElements(GL_TRIANGLES,caster.IndexCount,GL_UNSIGNED_INT,reinterpret_cast<const void*>(std::size_t(caster.FirstIndex)*sizeof(std::uint32_t)));
    }
    GlBindFramebuffer(GL_FRAMEBUFFER,0);
    if (!CheckGL("Draw current shadow receivers") || !PackVirtualShadowDepth(target.Depth,width,height,packed)) return false;
    return BuildVirtualShadowRequests({upload.Frame,{}},output,packed.Buffers[1]);
}
