// Material images use the ordinary immutable-per-draw descriptor set. The
// height block is a native push snapshot, so no later object can overwrite it.
std::array<VkShaderModule,2> ParallaxModules{};
void DestroyParallax()
{
    for (auto& module:ParallaxModules)
    { if (module && DestroyShaderModuleFunction) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
}
bool EnsureParallax()
{
    for (std::size_t stage=0;stage<ParallaxModules.size();++stage)
        if (!ParallaxModules[stage] && !CreateVulkanShaderModule(Device,CreateShaderModuleFunction,
            VulkanParallaxOcclusionShaders[stage],ParallaxModules[stage],LastError)) return false;
    return true;
}
bool BindParallaxDraw()
{
    ParallaxConstants constants;
    if (!BuildParallaxConstants(BoundPipeline.Parallax,BoundPipeline.ParallaxBoundsMinimum,BoundPipeline.ParallaxBoundsMaximum,constants))
        return Fail("POM requires finite nonempty local object bounds");
    CmdPushConstants(CommandBuffer,PipelineLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(constants),&constants); return true;
}
