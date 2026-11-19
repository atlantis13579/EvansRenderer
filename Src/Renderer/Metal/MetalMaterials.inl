std::array<ObjCObject,MaterialProgramCount> MaterialLibraries{};
std::unordered_map<std::uint32_t,ObjCObject> MaterialSurfacePipelines,MaterialGeometryPipelines;
ObjCObject MaterialLibrary(MaterialTechnique technique,unsigned model,bool geometry)
{
    if (unsigned(technique)>=MaterialTechniqueCount || model>=MaterialSurfaceModels) { Fail("Invalid Metal material program"); return nullptr; }
    auto& library=MaterialLibraries[MaterialProgramIndex(technique,model,false,geometry)];
    if (library) return library;
    const auto filename=std::string(MaterialTechniqueFile(technique))+".metal";
    std::string source; if (!LoadTechniqueShaderSource("Metal",filename.c_str(),source,LastError)) return nullptr;
    source.insert(0,"#define EVANS_MATERIAL_PROGRAM "+std::to_string(model)+"\n"+(geometry?"#define EVANS_MATERIAL_GBUFFER 1\n":""));
    ObjCObject error=nullptr;
    library=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
    if (!library) Fail("Metal material compilation failed: "+ErrorDescription(error));
    return library;
}
ObjCObject MaterialSurfacePipeline(MaterialTechnique technique,bool hdr,NSUInteger samples)
{
    const auto model=unsigned(BoundPipeline.Shading);
    constexpr unsigned TechniqueShift=16,SampleShift=4,HdrBit=1u<<8,WriteBit=1u<<9,BlendBit=1u<<10,WeightedBit=1u<<11;
    const auto key=(unsigned(technique)<<TechniqueShift)|model|(unsigned(samples)<<SampleShift)|(hdr?HdrBit:0u)|
        (BoundPipeline.ColorWrite?WriteBit:0u)|(BoundPipeline.BlendEnabled?BlendBit:0u)|(WeightedPass?WeightedBit:0u);
    auto& pipeline=MaterialSurfacePipelines[key]; if (pipeline) return pipeline;
    const auto library=MaterialLibrary(technique,model,false); if (!library) return nullptr;
    return pipeline=CreatePipeline(PassType::Color,BoundPipeline.ColorWrite,BoundPipeline.BlendEnabled,hdr,samples,WeightedPass,library);
}
ObjCObject MaterialGeometryPipeline(MaterialTechnique technique,NSUInteger samples)
{
    constexpr unsigned TechniqueShift=16,WriteBit=1u<<8;
    const auto key=(unsigned(technique)<<TechniqueShift)|unsigned(samples)|(BoundPipeline.ColorWrite?WriteBit:0u);
    auto& pipeline=MaterialGeometryPipelines[key]; if (pipeline) return pipeline;
    const auto library=MaterialLibrary(technique,0,true); if (!library) return nullptr;
    const auto filename=std::string(MaterialTechniqueFile(technique))+".metal";
    return pipeline=CreateTechniquePipeline(filename.c_str(),"DeferredGeometryVertex","DeferredGeometryFragment",samples,
        std::vector<NSUInteger>(GBufferAttachmentCount,PixelFormatRGBA16Float),true,BoundPipeline.ColorWrite?ColorWriteMaskAll:ColorWriteMaskNone,false,library);
}
void DestroyMaterials()
{
    for (auto& pair:MaterialSurfacePipelines) ReleaseObject(pair.second);
    for (auto& pair:MaterialGeometryPipelines) ReleaseObject(pair.second);
    MaterialSurfacePipelines.clear(); MaterialGeometryPipelines.clear();
    for (auto& library:MaterialLibraries) { ReleaseObject(library); library=nullptr; }
}
