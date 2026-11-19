// MTLPixelFormatRGBA32Uint; the native enum preserves all four 32-bit words.
static constexpr NSUInteger RayShadowRGBA32Uint=123,RayShadowTextureIndex=20;
std::shared_ptr<const RayShadowImage> FrameRayShadowImage,CachedRayShadowImage;
ObjCObject RayShadowTexture=nullptr,RayShadowDummyTexture=nullptr;

bool CreateRayShadowTexture(const RayShadowImage& image,ObjCObject& output)
{
    if (!image.Width || !image.Height || image.Texels.size()!=std::size_t(image.Width)*image.Height)
        return Fail("Metal ray-shadow image has invalid dimensions or payload");
    const auto candidate=CreateNativeTexture(int(image.Width),int(image.Height),RayShadowRGBA32Uint,
        TextureUsageShaderRead,"Evans ray-shadow immutable BVH");
    if (!candidate) return false;
    if (!UploadTexture(candidate,int(image.Width),int(image.Height),sizeof(RayShadowTexel),image.Texels.data()))
    { ReleaseObject(candidate); return false; }
    output=candidate; return true;
}
bool PrepareRayShadowImage(std::shared_ptr<const RayShadowImage> image)
{
    if (!RayShadowDummyTexture)
    {
        RayShadowImage empty; empty.Width=RayShadowHeaderTexels; empty.Height=1; empty.Texels.resize(RayShadowHeaderTexels);
        if (!CreateRayShadowTexture(empty,RayShadowDummyTexture)) return false;
    }
    if (image && image!=CachedRayShadowImage)
    {
        ObjCObject candidate=nullptr;
        if (!CreateRayShadowTexture(*image,candidate)) return false;
        ReleaseObject(RayShadowTexture); RayShadowTexture=candidate; CachedRayShadowImage=image;
    }
    FrameRayShadowImage=std::move(image); return true;
}
void BindRayShadowImage(ObjCObject encoder)
{
    const auto image=FrameRayShadowImage?RayShadowTexture:RayShadowDummyTexture;
    Send<void,ObjCObject,NSUInteger>(encoder,"setFragmentTexture:atIndex:",image,RayShadowTextureIndex);
    Send<void,ObjCObject,NSUInteger>(encoder,"setVertexTexture:atIndex:",image,RayShadowTextureIndex);
}
void DestroyRayShadowImage()
{
    FrameRayShadowImage.reset(); CachedRayShadowImage.reset();
    ReleaseObject(RayShadowTexture); ReleaseObject(RayShadowDummyTexture);
    RayShadowTexture=RayShadowDummyTexture=nullptr;
}
