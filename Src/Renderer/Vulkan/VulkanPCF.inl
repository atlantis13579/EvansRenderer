std::array<VkShaderModule,13> PCFFragments{};
std::array<std::array<VkShaderModule,2>,3> PCFScreenModules{};
std::array<VkPipeline,3> PCFScreenPipelines{};
bool EnsurePCFFragment(unsigned program,bool weighted=false)
{
    constexpr unsigned ProgramCount=8,SurfaceCount=5;
    if (program>=ProgramCount || (weighted && program>=SurfaceCount)) return Fail("Invalid Vulkan PCF program");
    auto& module=PCFFragments[weighted?ProgramCount+program:program];
    if (module) return true;
    const auto definitions=std::string("EVANS_PCF_PROGRAM=")+std::to_string(program)+(weighted?";EVANS_WEIGHTED_TRANSPARENCY=1":"");
    return CreateVulkanShaderModule(Device,CreateShaderModuleFunction,
        {"PCF.glsl",VulkanShaderStage::Fragment,definitions.c_str()},module,LastError);
}
VkPipeline PCFScreen(unsigned screen)
{
    if (screen>=PCFScreenPipelines.size()) return 0;
    const auto definitions=std::string("EVANS_PCF_PROGRAM=")+std::to_string(5+screen);
    const auto vertexDefinitions=definitions+";EVANS_VERTEX_STAGE=1";
    const std::array<VulkanShaderProgram,2> shaders{{
        {"PCF.glsl",VulkanShaderStage::Vertex,vertexDefinitions.c_str()},
        {"PCF.glsl",VulkanShaderStage::Fragment,definitions.c_str()}}};
    return CreateFullscreenPipeline(shaders.data(),PCFScreenModules[screen],screen==2?VolumeLightRenderPass:screen==1?Multisample.LightingPass:HdrRenderPass,
        PCFScreenPipelines[screen],screen==2?VolumeLightPipelineLayout:DeferredPipelineLayout,
        FullscreenBlend::Replace,false,screen==2?1:FrameSampleCount,screen==1);
}
void DestroyPCFPrograms()
{
    for (auto& pipeline:PCFScreenPipelines)
    { if (pipeline) DestroyPipelineFunction(Device,pipeline,nullptr); pipeline=0; }
    for (auto& modules:PCFScreenModules) for (auto& module:modules)
    { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
    for (auto& module:PCFFragments)
    { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; }
}
