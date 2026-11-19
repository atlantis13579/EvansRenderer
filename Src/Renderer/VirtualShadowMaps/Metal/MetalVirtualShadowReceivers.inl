// Raster, depth packing and page marking consume native resources. Shared
// storage on Apple devices does not imply a CPU reconstruction or depth copy.
bool BuildVirtualShadowReceiverRequests(const VirtualShadowReceiverUpload& upload,VirtualShadowRequestResult& output)
{
    VirtualShadowRequestConstants request;
    if (!BuildVirtualShadowRequestConstants(upload.Frame,request,LastError) || !EnsureVirtualShadowAtlas()) return false;
    const auto width=upload.Frame.Width,height=upload.Frame.Height;
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
    struct Resources { ObjCObject Depth=nullptr,Packed=nullptr; ~Resources(){ReleaseObject(Packed); ReleaseObject(Depth);} } resources;
    const auto descriptor=Send<ObjCObject,NSUInteger,NSUInteger,NSUInteger,ObjCBool>(GetClass("MTLTextureDescriptor"),
        "texture2DDescriptorWithPixelFormat:width:height:mipmapped:",PixelFormatDepth32Float,width,height,ObjCNo);
    Send<void,NSUInteger>(descriptor,"setStorageMode:",StorageModePrivate); Send<void,NSUInteger>(descriptor,"setUsage:",TextureUsageRenderTarget|TextureUsageShaderRead);
    resources.Depth=Send<ObjCObject,ObjCObject>(Device,"newTextureWithDescriptor:",descriptor); if (!resources.Depth) return Fail("Metal receiver depth allocation failed");
    const auto command=Send<ObjCObject>(CommandQueue,"commandBuffer"); if (!command) return Fail("Metal receiver commands are unavailable");
    const auto pass=Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"),"renderPassDescriptor"),attachment=Send<ObjCObject>(pass,"depthAttachment");
    Send<void,ObjCObject>(attachment,"setTexture:",resources.Depth); Send<void,NSUInteger>(attachment,"setLoadAction:",LoadActionClear);
    Send<void,NSUInteger>(attachment,"setStoreAction:",StoreActionStore); Send<void,double>(attachment,"setClearDepth:",1);
    const auto encoder=Send<ObjCObject,ObjCObject>(command,"renderCommandEncoderWithDescriptor:",pass);
    if (!encoder) return Fail("Metal receiver depth encoder is unavailable");
    Send<void,ObjCObject>(encoder,"setRenderPipelineState:",VirtualAtlasRaster); Send<void,ObjCObject>(encoder,"setDepthStencilState:",VirtualAtlasDepthState);
    Send<void,NSUInteger>(encoder,"setTriangleFillMode:",TriangleFillModeFill);
    const MTLViewport viewport{0,0,double(width),double(height),0,1}; const VirtualAtlasScissor scissor{0,0,width,height};
    Send<void,MTLViewport>(encoder,"setViewport:",viewport); Send<void,VirtualAtlasScissor>(encoder,"setScissorRect:",scissor);
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentSamplerState:atIndex:",SurfaceSampler,0);
    for (std::size_t draw=0;draw<draws.size();++draw)
    {
        const auto& caster=upload.Receivers[draw];
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setVertexBytes:length:atIndex:",&draws[draw],sizeof(draws[draw]),1);
        Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&draws[draw],sizeof(draws[draw]),1);
        Send<void,NSUInteger>(encoder,"setCullMode:",caster.BackFaceCulling?CullModeBack:CullModeNone);
        Send<void,NSUInteger>(encoder,"setFrontFacingWinding:",RealtimeWorldIsMirrored(caster.World)?WindingClockwise:WindingCounterClockwise);
        Send<void,ObjCObject,NSUInteger,NSUInteger>(encoder,"setVertexBuffer:offset:atIndex:",Buffers.at(caster.Vertices.Value).Buffer,0,0);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",caster.Diffuse?Textures.at(caster.Diffuse.Value).Texture:nullptr,0);
        Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",caster.Opacity?Textures.at(caster.Opacity.Value).Texture:nullptr,1);
        Send<void,NSUInteger,NSUInteger,NSUInteger,ObjCObject,NSUInteger>(encoder,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
            PrimitiveTypeTriangle,caster.IndexCount,IndexTypeUInt32,Buffers.at(caster.Indices.Value).Buffer,NSUInteger(caster.FirstIndex)*sizeof(std::uint32_t));
    }
    Send<void>(encoder,"endEncoding");
    if (!EncodeVirtualDepthPack(command,resources.Depth,width,height,resources.Packed)) return false;
    Send<void>(command,"commit"); Send<void>(command,"waitUntilCompleted");
    if (Send<NSUInteger>(command,"status")==CommandBufferStatusError) return Fail("Metal receiver depth failed: "+ErrorDescription(Send<ObjCObject>(command,"error")));
    return BuildVirtualShadowRequests({upload.Frame,{}},output,resources.Packed);
}
