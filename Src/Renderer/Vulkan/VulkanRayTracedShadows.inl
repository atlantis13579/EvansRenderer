std::shared_ptr<const RayShadowImage> FrameRayShadowImage,CachedRayShadowImage;
RawImage RayShadowTexture,RayShadowDummyTexture;
VkSampler RayShadowSampler=0;

bool CreateRayShadowTexture(const RayShadowImage& image,RawImage& output)
{
    if (!image.Width || !image.Height || image.Texels.size()!=std::size_t(image.Width)*image.Height)
        return Fail("Vulkan ray-shadow image has invalid dimensions or payload");
    RawImage candidate;
    if (!CreateRawImage(image.Width,image.Height,VK_FORMAT_R32G32B32A32_UINT,VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,candidate)) return false;
    // Complete the immutable upload before frame recording. An aborted graphics
    // frame cannot leave its CPU cache pointing at unsubmitted image contents.
    if (!InitializeImage(candidate,image.Texels.data(),image.Texels.size()*sizeof(RayShadowTexel)))
    { DestroyRawImage(candidate); return false; }
    output=candidate; return true;
}
bool PrepareRayShadowImage(std::shared_ptr<const RayShadowImage> image)
{
    if (!RayShadowSampler)
    {
        VkSamplerCreateInfo create{}; create.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        create.magFilter=create.minFilter=VK_FILTER_NEAREST; create.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
        create.addressModeU=create.addressModeV=create.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        create.maxAnisotropy=1; create.maxLod=0;
        if (!Check(CreateSamplerFunction(Device,&create,nullptr,&RayShadowSampler),"Create Vulkan ray-shadow integer sampler")) return false;
    }
    if (!RayShadowDummyTexture.Image)
    {
        RayShadowImage empty; empty.Width=RayShadowHeaderTexels; empty.Height=1; empty.Texels.resize(RayShadowHeaderTexels);
        if (!CreateRayShadowTexture(empty,RayShadowDummyTexture)) return false;
    }
    if (image && image!=CachedRayShadowImage)
    {
        RawImage candidate; if (!CreateRayShadowTexture(*image,candidate)) return false;
        DestroyRawImage(RayShadowTexture); RayShadowTexture=candidate; CachedRayShadowImage=image;
    }
    FrameRayShadowImage=std::move(image); return true;
}
VkDescriptorImageInfo RayShadowImageDescriptor() const
{
    const auto& image=FrameRayShadowImage?RayShadowTexture:RayShadowDummyTexture;
    return {RayShadowSampler,image.View,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
}
void DestroyRayShadowImage()
{
    FrameRayShadowImage.reset(); CachedRayShadowImage.reset(); DestroyRawImage(RayShadowTexture); DestroyRawImage(RayShadowDummyTexture);
    if (RayShadowSampler) DestroySamplerFunction(Device,RayShadowSampler,nullptr);
    RayShadowSampler=0;
}

std::array<VkShaderModule,13> RayShadowFragments{};
std::array<std::array<VkShaderModule,2>,3> RayShadowScreenModules{};
std::array<VkPipeline,3> RayShadowScreenPipelines{};
bool EnsureRayShadowFragment(unsigned program,bool weighted=false)
{
    constexpr unsigned ProgramCount=8,SurfaceCount=5;
    if (program>=ProgramCount || (weighted && program>=SurfaceCount)) return Fail("Invalid Vulkan ray-shadow program");
    auto& module=RayShadowFragments[weighted?ProgramCount+program:program];
    if (module) return true;
    const auto definitions=std::string("EVANS_RAY_SHADOW_PROGRAM=")+std::to_string(program)+(weighted?";EVANS_WEIGHTED_TRANSPARENCY=1":"");
    return CreateVulkanShaderModule(Device,CreateShaderModuleFunction,
        {"RayTracedShadows.glsl",VulkanShaderStage::Fragment,definitions.c_str()},module,LastError);
}
VkPipeline RayShadowScreen(unsigned screen)
{
    if (screen>=RayShadowScreenPipelines.size()) return 0;
    const auto definitions=std::string("EVANS_RAY_SHADOW_PROGRAM=")+std::to_string(5+screen);
    const auto vertexDefinitions=definitions+";EVANS_VERTEX_STAGE=1";
    const std::array<VulkanShaderProgram,2> shaders{{
        {"RayTracedShadows.glsl",VulkanShaderStage::Vertex,vertexDefinitions.c_str()},
        {"RayTracedShadows.glsl",VulkanShaderStage::Fragment,definitions.c_str()}}};
    return CreateFullscreenPipeline(shaders.data(),RayShadowScreenModules[screen],screen==2?VolumeLightRenderPass:screen==1?Multisample.LightingPass:HdrRenderPass,
        RayShadowScreenPipelines[screen],screen==2?VolumeLightPipelineLayout:DeferredPipelineLayout,
        FullscreenBlend::Replace,false,screen==2?1:FrameSampleCount,screen==1);
}
void DestroyRayShadowPrograms()
{
    for (auto& pipeline:RayShadowScreenPipelines)
    { if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    for (auto& modules:RayShadowScreenModules) for (auto& module:modules)
    { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    for (auto& module:RayShadowFragments)
    { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
}
