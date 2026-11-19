ObjCObject ParallaxLibrary=nullptr;
std::unordered_map<unsigned,ObjCObject> ParallaxPipelines;
void DestroyParallax()
{
    for (auto pair:ParallaxPipelines) ReleaseObject(pair.second);
    ParallaxPipelines.clear(); ReleaseObject(ParallaxLibrary); ParallaxLibrary=nullptr;
}
ObjCObject ParallaxMaterialPipeline(unsigned samples,bool colorWrite)
{
    const unsigned key=(samples<<1)|(colorWrite?1u:0u);
    const auto found=ParallaxPipelines.find(key); if (found!=ParallaxPipelines.end()) return found->second;
    if (!ParallaxLibrary)
    {
        std::string source; if (!LoadTechniqueShaderSource("Metal","ParallaxOcclusionMapping.metal",source,LastError)) return nullptr;
        ObjCObject options=PositionInvariantCompileOptions(),error=nullptr;
        if (!SupportsPositionInvariance(options)) { ReleaseObject(options); Fail("Metal POM requires position-invariance compilation"); return nullptr; }
        ParallaxLibrary=Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),options,&error);
        ReleaseObject(options); if (!ParallaxLibrary) { Fail("Metal POM compilation failed: "+ErrorDescription(error)); return nullptr; }
    }
    const std::vector<NSUInteger> formats(GBufferAttachmentCount,PixelFormatRGBA16Float);
    ObjCObject pipeline=CreateTechniquePipeline("ParallaxOcclusionMapping.metal","VSMain","PSMain",samples,formats,true,
        colorWrite?ColorWriteMaskAll:0,false,ParallaxLibrary);
    if (pipeline) ParallaxPipelines.emplace(key,pipeline); return pipeline;
}
bool BindParallaxDraw()
{
    ParallaxConstants constants;
    if (!BuildParallaxConstants(BoundPipeline.Parallax,BoundPipeline.ParallaxBoundsMinimum,BoundPipeline.ParallaxBoundsMaximum,constants))
        return Fail("POM requires finite nonempty local object bounds");
    Send<void,const void*,NSUInteger,NSUInteger>(RenderEncoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),2); return true;
}
