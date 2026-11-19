std::array<VkShaderModule,MaterialProgramCount> MaterialFragments{};
bool EnsureMaterialFragment(MaterialTechnique technique,unsigned model,bool weighted,bool geometry)
{
    if (unsigned(technique)>=MaterialTechniqueCount || model>=MaterialSurfaceModels) return Fail("Invalid Vulkan material program");
    auto& module=MaterialFragments[MaterialProgramIndex(technique,model,weighted,geometry)];
    if (module) return true;
    const auto file=std::string(MaterialTechniqueFile(technique))+".glsl";
    const auto definitions=std::string("EVANS_MATERIAL_PROGRAM=")+std::to_string(model)+
        (geometry?";EVANS_MATERIAL_GBUFFER=1":"")+(weighted?";EVANS_WEIGHTED_TRANSPARENCY=1":"");
    return CreateVulkanShaderModule(Device,CreateShaderModuleFunction,
        {file.c_str(),VulkanShaderStage::Fragment,definitions.c_str()},module,LastError);
}
void DestroyMaterials()
{ for (auto& module:MaterialFragments) { if (module) DestroyShaderModuleFunction(Device,module,nullptr); module=0; } }
