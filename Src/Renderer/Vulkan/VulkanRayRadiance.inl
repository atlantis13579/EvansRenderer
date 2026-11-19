// Immutable scene radiance storage is shared by diffuse GI and specular
// reflections. Each effect retains independent shaders, constants and targets.
static constexpr std::size_t RayRadianceAtlasSlot=RaySceneShadingBufferCount-1;
RayRadianceFrame FrameRayRadiance;
Techniques::Settings FrameRayRadianceSettings;
std::shared_ptr<const RaySceneShadingUpload> UploadedRayShading;
std::shared_ptr<const SceneLightCollection> UploadedRayLights;
std::array<RawBuffer,RayRadianceBufferCount> RayRadianceBuffers{};
RawImage RayRadianceColorAtlas;
static constexpr std::uint32_t RayRadianceStorageCount=RaySceneBufferCount+RayRadianceBufferCount-1;

void DestroyRayRadiance()
{
    DestroyRawImage(RayRadianceColorAtlas);
    for (auto& buffer:RayRadianceBuffers) DestroyRawBuffer(buffer);
    UploadedRayShading.reset(); UploadedRayLights.reset(); FrameRayRadiance={};
}

bool EnsureRayRadiance()
{
    const auto& frame=FrameRayRadiance;
    if (!frame.Geometry || frame.Geometry!=FrameRayScene || !frame.Shading || !frame.Lights)
        return Fail("Vulkan ray radiance requires matched geometry, material and light snapshots");
    if (DeviceLimits.maxPerStageDescriptorStorageBuffers<RayRadianceStorageCount ||
        DeviceLimits.maxDescriptorSetStorageBuffers<RayRadianceStorageCount ||
        DeviceLimits.maxFragmentCombinedOutputResources<RayRadianceStorageCount+1)
        return Fail("Vulkan ray radiance requires seven fragment storage buffers and one color output");
    if (!EnsureRayScene()) return false;
    const bool materials=frame.Shading!=UploadedRayShading,lights=frame.Lights!=UploadedRayLights;
    if (!materials && !lights) return true;
    if (materials && !ValidateRaySceneShadingUpload(*frame.Geometry,*frame.Shading,LastError)) return false;
    const auto data=GetRayRadianceBufferData(frame);
    for (std::size_t slot=0;slot<data.size();++slot) if (slot!=RayRadianceAtlasSlot && data[slot].Bytes>DeviceLimits.maxStorageBufferRange)
        return Fail("Vulkan ray shading exceeds maxStorageBufferRange");
    std::array<RawBuffer,RayRadianceBufferCount> replacements{};
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (slot==RayRadianceAtlasSlot || !(slot<RaySceneShadingBufferCount ? materials : lights)) continue;
        if (!CreateRawBuffer(data[slot].Bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,replacements[slot]))
        { for (auto& buffer:replacements) DestroyRawBuffer(buffer); return false; }
        std::memcpy(replacements[slot].Mapped,data[slot].Data,data[slot].Bytes);
    }
    RawImage atlas;
    if (materials)
    {
        const auto count=std::max<std::size_t>(1,frame.Shading->Texels.size());
        const auto width=std::min<std::size_t>(count,DeviceLimits.maxImageDimension2D);
        const auto height=width ? (count+width-1)/width : 0;
        if (!height || height>DeviceLimits.maxImageDimension2D)
        {
            for (auto& buffer:replacements) DestroyRawBuffer(buffer);
            return Fail("Vulkan reflection color atlas exceeds maxImageDimension2D");
        }
        RawBuffer staging;
        const auto bytes=width*height*sizeof(Float4);
        if (!CreateRawImage(static_cast<std::uint32_t>(width),static_cast<std::uint32_t>(height),VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,atlas) ||
            !CreateRawBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,staging))
        {
            DestroyRawImage(atlas); DestroyRawBuffer(staging);
            for (auto& buffer:replacements) DestroyRawBuffer(buffer); return false;
        }
        std::memcpy(staging.Mapped,data[RayRadianceAtlasSlot].Data,data[RayRadianceAtlasSlot].Bytes);
        std::memset(static_cast<std::uint8_t*>(staging.Mapped)+data[RayRadianceAtlasSlot].Bytes,0,bytes-data[RayRadianceAtlasSlot].Bytes);
        // Immutable atlas upload finishes before frame recording. Cancellation
        // cannot discard this copy while retaining its published CPU snapshot.
        bool uploaded=BeginImmediate();
        if (uploaded)
        {
            TransitionImage(CommandBuffer,atlas,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
            copy.imageExtent={atlas.Width,atlas.Height,1};
            CmdCopyBufferToImage(CommandBuffer,staging.Buffer,atlas.Image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
            TransitionImage(CommandBuffer,atlas,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            uploaded=EndImmediate();
        }
        DestroyRawBuffer(staging);
        if (!uploaded)
        { DestroyRawImage(atlas); for (auto& buffer:replacements) DestroyRawBuffer(buffer); return false; }
    }
    for (std::size_t slot=0;slot<replacements.size();++slot) if (replacements[slot].Buffer)
    { DestroyRawBuffer(RayRadianceBuffers[slot]); RayRadianceBuffers[slot]=replacements[slot]; }
    if (atlas.Image) { DestroyRawImage(RayRadianceColorAtlas); RayRadianceColorAtlas=atlas; }
    UploadedRayShading=frame.Shading; UploadedRayLights=frame.Lights; return true;
}
