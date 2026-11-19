// Tracked Metal textures order raster writes before material reads across two
// render encoders. The immutable shared triangle buffer survives Present's wait.
static constexpr NSUInteger VisibilityIntegerFormat = 53; // MTLPixelFormatR32Uint.
std::array<ObjCObject, 2> VisibilityImages{}, VisibilityPipelines{};
ObjCObject VisibilityGeometry = nullptr, VisibilityLibrary = nullptr;
int VisibilityWidth = 0, VisibilityHeight = 0;
bool VisibilitySupported = false;
std::shared_ptr<const VisibilityScene> UploadedVisibility;

void DestroyVisibilityBuffer()
{
    DestroyMeshShaders();
    DestroyNaniteMaterials();
    DestroyNaniteFrameRaster();
    DestroyNaniteVisibilityGeometry();
    for (auto image : VisibilityImages) ReleaseObject(image);
    for (auto pipeline : VisibilityPipelines) ReleaseObject(pipeline);
    ReleaseObject(VisibilityGeometry); ReleaseObject(VisibilityLibrary);
    VisibilityImages = {}; VisibilityPipelines = {}; VisibilityGeometry = VisibilityLibrary = nullptr;
    VisibilityWidth = VisibilityHeight = 0; UploadedVisibility.reset();
}
bool ProbeVisibilityBuffer()
{
    const auto previous = LastError;
    auto id = CreateNativeTexture(1, 1, VisibilityIntegerFormat, TextureUsageShaderRead | TextureUsageRenderTarget, "Visibility ID probe");
    auto bary = CreateNativeTexture(1, 1, PixelFormatRG32Float, TextureUsageShaderRead | TextureUsageRenderTarget, "Visibility weights probe");
    const bool valid = id && bary; ReleaseObject(id); ReleaseObject(bary); LastError = previous;
    return valid;
}
bool PrepareVisibilityBuffer(const std::shared_ptr<const VisibilityScene>& scene)
{
    if (!VisibilitySupported || !scene) return Fail("Metal visibility requires renderable integer/float targets");
    if (scene->HasNativeGeometry())
    {
        if (!PrepareNaniteVisibilityGeometry(*scene)) return false;
    }
    else if (UploadedVisibility != scene)
    {
        if (!ValidateVisibilityScene(*scene, LastError)) return false;
        std::vector<VisibilityTriangle> bounds;
        if (!BuildMeshShaderBounds(*scene,bounds,LastError)) return false;
        const auto geometryBytes=scene->Triangles.size()*sizeof(VisibilityTriangle);
        const auto bytes=std::max(scene->Triangles.size()+bounds.size(),std::size_t{1})*sizeof(VisibilityTriangle);
        auto candidate = Send<ObjCObject, NSUInteger, NSUInteger>(Device, "newBufferWithLength:options:", bytes, 0);
        if (!candidate) return Fail("Metal visibility geometry allocation failed");
        auto* mapped = Send<void*>(candidate, "contents");
        if (!mapped) { ReleaseObject(candidate); return Fail("Metal visibility geometry is not host visible"); }
        if (geometryBytes) std::memcpy(mapped,scene->Triangles.data(),geometryBytes); else std::memset(mapped,0,bytes);
        if (!bounds.empty()) std::memcpy(static_cast<std::byte*>(mapped)+geometryBytes,bounds.data(),bounds.size()*sizeof(VisibilityTriangle));
        ReleaseObject(VisibilityGeometry); VisibilityGeometry = candidate; UploadedVisibility = scene;
    }
    if (VisibilityWidth != CreateInfo.Width || VisibilityHeight != CreateInfo.Height)
    {
        for (auto& image : VisibilityImages) { ReleaseObject(image); image = nullptr; }
        VisibilityWidth = VisibilityHeight = 0;
        VisibilityImages[0] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, VisibilityIntegerFormat,
            TextureUsageShaderRead | TextureUsageRenderTarget, "Visibility triangle references");
        VisibilityImages[1] = CreateNativeTexture(CreateInfo.Width, CreateInfo.Height, PixelFormatRG32Float,
            TextureUsageShaderRead | TextureUsageRenderTarget, "Visibility barycentrics");
        if (!VisibilityImages[0] || !VisibilityImages[1]) return false;
        VisibilityWidth = CreateInfo.Width; VisibilityHeight = CreateInfo.Height;
    }
    if (!VisibilityLibrary)
    {
        std::string source;
        if (!LoadTechniqueShaderSource("Metal", "VisibilityBuffer.metal", source, LastError)) return false;
        auto options = Send<ObjCObject>(GetClass("MTLCompileOptions"), "new");
        Send<void, ObjCBool>(options, "setFastMathEnabled:", ObjCNo);
        ObjCObject error = nullptr;
        VisibilityLibrary = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options);
        if (!VisibilityLibrary) return Fail("Metal visibility source compilation failed: " + ErrorDescription(error));
    }
    if (!VisibilityPipelines[0]) VisibilityPipelines[0] = CreateTechniquePipeline("VisibilityBuffer.metal",
        "EvansVisibilityRasterVertex", "EvansVisibilityRasterFragment", 1, {VisibilityIntegerFormat, PixelFormatRG32Float},
        true, ColorWriteMaskAll, false, VisibilityLibrary);
    if (!VisibilityPipelines[1]) VisibilityPipelines[1] = CreateTechniquePipeline("VisibilityBuffer.metal",
        "EvansVisibilityResolveVertex", "EvansVisibilityResolveFragment", 1,
        std::vector<NSUInteger>(GBufferAttachmentCount, PixelFormatRGBA16Float), false, ColorWriteMaskAll, false, VisibilityLibrary);
    return VisibilityPipelines[0] && VisibilityPipelines[1];
}
bool DrawVisibilityBuffer(const VisibilityBufferDesc& frame)
{
    if (RenderEncoder) { Send<void>(RenderEncoder, "endEncoding"); RenderEncoder = nullptr; }
    const MTLViewport viewport{0, 0, double(CreateInfo.Width), double(CreateInfo.Height), 0, 1};
    const auto begin = [&](bool resolve)
    {
        auto descriptor = Send<ObjCObject>(GetClass("MTLRenderPassDescriptor"), "renderPassDescriptor");
        auto colors = Send<ObjCObject>(descriptor, "colorAttachments");
        const std::size_t count = resolve ? GBufferAttachmentCount : 2;
        for (std::size_t index = 0; index < count; ++index)
        {
            auto attachment = Send<ObjCObject, NSUInteger>(colors, "objectAtIndexedSubscript:", index);
            Send<void, ObjCObject>(attachment, "setTexture:", resolve ? GBufferTextures[index] : VisibilityImages[index]);
            Send<void, NSUInteger>(attachment, "setLoadAction:", resolve ? LoadActionLoad : LoadActionClear);
            Send<void, NSUInteger>(attachment, "setStoreAction:", StoreActionStore);
            if (!resolve) Send<void, MTLClearColor>(attachment, "setClearColor:", MTLClearColor{0, 0, 0, 0});
        }
        if (!resolve)
        {
            auto depth = Send<ObjCObject>(descriptor, "depthAttachment");
            Send<void, ObjCObject>(depth, "setTexture:", DepthTexture);
            Send<void, NSUInteger>(depth, "setLoadAction:", LoadActionLoad); Send<void, NSUInteger>(depth, "setStoreAction:", StoreActionStore);
        }
        RenderEncoder = Send<ObjCObject, ObjCObject>(CommandBuffer, "renderCommandEncoderWithDescriptor:", descriptor);
        if (!RenderEncoder) return Fail("Metal visibility render encoder creation failed");
        Send<void, MTLViewport>(RenderEncoder, "setViewport:", viewport);
        if (resolve || !frame.MeshShaders) Send<void,ObjCObject>(RenderEncoder,"setRenderPipelineState:",VisibilityPipelines[resolve?1:0]);
        Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder, "setVertexBuffer:offset:atIndex:", VisibilityGeometry, 0, 1);
        Send<void, ObjCObject, NSUInteger, NSUInteger>(RenderEncoder, "setFragmentBuffer:offset:atIndex:", VisibilityGeometry, 0, 1);
        if (!resolve && frame.MeshShaders)
        {
            Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setMeshBuffer:offset:atIndex:",VisibilityGeometry,0,1);
            Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setObjectBuffer:offset:atIndex:",VisibilityGeometry,0,1);
        }
        Send<void, ObjCObject, NSUInteger>(RenderEncoder, "setFragmentSamplerState:atIndex:", MaterialSampler(), 0);
        if (resolve) for (NSUInteger image = 0; image < 2; ++image)
            Send<void, ObjCObject, NSUInteger>(RenderEncoder, "setFragmentTexture:atIndex:", VisibilityImages[image], image);
        return true;
    };
    const auto bind = [&](const VisibilityMaterialDraw& draw)
    {
        ObjCObject images[] {DummyColorTexture, DummyColorTexture, DummyColorTexture};
        const TextureHandle handles[] {draw.Pipeline.DiffuseTexture, draw.Pipeline.NormalTexture, draw.Pipeline.OpacityTexture};
        for (std::size_t image = 0; image < 3; ++image) if (handles[image])
        {
            const auto found = Textures.find(handles[image].Value);
            if (found == Textures.end() || found->second.Description.Format != TextureFormat::RGBA8) return Fail("Metal visibility material texture is invalid");
            images[image] = found->second.Texture;
        }
        VisibilityConstants constants;
        if (!BuildVisibilityConstants(frame.ViewProjection, draw.Pipeline, CreateInfo.Width, CreateInfo.Height, frame.Scene->TriangleCount(),
            frame.Scene->Draws[draw.Range], bool(handles[0]), bool(handles[1]), bool(handles[2]), constants)) return Fail("Invalid Metal visibility material");
        constants.MaterialTiles=MaterialPlan.Resolve(unsigned(&draw-frame.Draws.data()));
        if (frame.MeshShaders)
        {
            Send<void,const void*,NSUInteger,NSUInteger>(RenderEncoder,"setMeshBytes:length:atIndex:",&constants,sizeof(constants),0);
            Send<void,const void*,NSUInteger,NSUInteger>(RenderEncoder,"setObjectBytes:length:atIndex:",&constants,sizeof(constants),0);
        }
        Send<void, const void*, NSUInteger, NSUInteger>(RenderEncoder, "setVertexBytes:length:atIndex:", &constants, sizeof(constants), 0);
        Send<void, const void*, NSUInteger, NSUInteger>(RenderEncoder, "setFragmentBytes:length:atIndex:", &constants, sizeof(constants), 0);
        for (NSUInteger image = 0; image < 3; ++image) Send<void, ObjCObject, NSUInteger>(RenderEncoder, "setFragmentTexture:atIndex:", images[image], image + 2);
        return true;
    };
    if (frame.NaniteRaster) { if (!DrawNaniteFrameRaster(frame)) return false; }
    else
    {
        if (!begin(false)) return false;
        for (const auto& draw : frame.Draws)
        {
            if (!bind(draw)) return false;
            const auto& pipeline = draw.Pipeline; const auto& range = frame.Scene->Draws[draw.Range];
            Send<void, ObjCObject>(RenderEncoder, "setDepthStencilState:", !pipeline.DepthTest ? DepthDisabledState : pipeline.DepthWrite ? DepthWriteState : DepthReadState);
            Send<void, NSUInteger>(RenderEncoder, "setCullMode:", pipeline.BackFaceCulling ? CullModeBack : CullModeNone);
            Send<void, NSUInteger>(RenderEncoder, "setFrontFacingWinding:", range.Mirrored ? WindingClockwise : WindingCounterClockwise);
            Send<void, NSUInteger>(RenderEncoder, "setTriangleFillMode:", pipeline.Fill == FillMode::Wireframe ? TriangleFillModeLines : TriangleFillModeFill);
            if (frame.MeshShaders)
            {
                const bool task=UseMeshTaskStage(frame.MeshDispatch,range.TriangleCount,true);
                Send<void,ObjCObject>(RenderEncoder,"setRenderPipelineState:",MeshPipelines[task?3:0]);
                DrawMeshGeometry(RenderEncoder,range.TriangleCount,task);
                if (task) ++PendingFrame.Statistics.MeshShaderObjectDispatchCount;
            }
            else Send<void,NSUInteger,NSUInteger,NSUInteger>(RenderEncoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,range.TriangleCount*VisibilityTriangleCorners);
        }
        Send<void>(RenderEncoder, "endEncoding"); RenderEncoder = nullptr;
    }
    if (!ClassifyNaniteMaterials() || !begin(true)) return false;
    Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setVertexBuffer:offset:atIndex:",MaterialPlan.Active()?MaterialWork->Buffer:VisibilityGeometry,0,3);
    Send<void,ObjCObject,NSUInteger,NSUInteger>(RenderEncoder,"setFragmentBuffer:offset:atIndex:",MaterialPlan.Active()?MaterialMap->Buffer:VisibilityGeometry,0,2);
    Send<void, ObjCObject>(RenderEncoder, "setDepthStencilState:", DepthDisabledState);
    Send<void, NSUInteger>(RenderEncoder, "setCullMode:", CullModeNone);
    Send<void, NSUInteger>(RenderEncoder, "setTriangleFillMode:", TriangleFillModeFill);
    const auto resolveCount=MaterialPlan.Active()?MaterialPlan.Representatives.size():frame.Draws.size();
    for (unsigned bin=0;bin<resolveCount;++bin)
    {
        if (!bind(frame.Draws[MaterialPlan.Active()?MaterialPlan.Representatives[bin]:bin])) return false;
        if (MaterialPlan.Active()) Send<void,NSUInteger,ObjCObject,NSUInteger>(RenderEncoder,"drawPrimitives:indirectBuffer:indirectBufferOffset:",
            PrimitiveTypeTriangle,MaterialWork->Buffer,bin*NaniteMaterialBinWords*sizeof(unsigned));
        else Send<void,NSUInteger,NSUInteger,NSUInteger>(RenderEncoder,"drawPrimitives:vertexStart:vertexCount:",PrimitiveTypeTriangle,0,3);
    }
    // EndPass closes this final encoder and publishes completed material inputs.
    return true;
}
