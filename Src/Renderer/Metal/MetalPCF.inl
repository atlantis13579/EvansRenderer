std::array<ObjCObject,8> PCFLibraries{};
std::unordered_map<std::uint32_t,ObjCObject> PCFSurfacePipelines,PCFScreenPipelines;
        ObjCObject PCFLibrary(std::size_t program)
        {
            auto& library = PCFLibraries[program];
            if (library) return library;
            std::string source;
            if (!LoadTechniqueShaderSource("Metal", "PCF.metal", source, LastError)) return nullptr;
            source.insert(0, "#define EVANS_PCF_PROGRAM " + std::to_string(program) + "\n");
            ObjCObject error = nullptr;
            library = Evans::Renderer::CompileCachedMetalLibrary(Device,MakeString(source.c_str()),nullptr,&error);
            if (!library) Fail("Metal PCF shader compilation failed: " + ErrorDescription(error));
            return library;
        }

        ObjCObject PCFSurfacePipeline(bool hdr, NSUInteger samples)
        {
            const auto model = static_cast<std::uint32_t>(BoundPipeline.Shading);
            constexpr std::uint32_t SampleShift = 4, HdrBit = 1u << 8, WriteBit = 1u << 9,
                BlendBit = 1u << 10, WeightedBit = 1u << 11;
            const auto key = model | (std::uint32_t(samples) << SampleShift) | (hdr ? HdrBit : 0u) |
                (BoundPipeline.ColorWrite ? WriteBit : 0u) | (BoundPipeline.BlendEnabled ? BlendBit : 0u) |
                (WeightedPass ? WeightedBit : 0u);
            auto& pipeline = PCFSurfacePipelines[key];
            if (pipeline) return pipeline;
            const auto library = PCFLibrary(model);
            if (!library) return nullptr;
            pipeline = CreatePipeline(PassType::Color, BoundPipeline.ColorWrite, BoundPipeline.BlendEnabled,
                hdr, samples, WeightedPass, library);
            return pipeline;
        }

        ObjCObject PCFScreenPipeline(std::uint32_t program, NSUInteger samples = 1)
        {
            constexpr std::uint32_t ProgramShift = 16;
            auto& pipeline = PCFScreenPipelines[(program << ProgramShift) | std::uint32_t(samples)];
            if (pipeline) return pipeline;
            const auto library = PCFLibrary(program);
            if (!library) return nullptr;
            const bool volume = program == 7;
            pipeline = CreateTechniquePipeline("PCF.metal", volume ? "VolumeVertexMain" : "DeferredLightingVertex",
                volume ? "VolumeFragmentMain" : program == 6 ? "DeferredLightingSampleFragment" : "DeferredLightingFragment",
                samples, {volume ? PixelFormatRGBA32Float : PixelFormatRGBA16Float}, false,
                ColorWriteMaskAll, false, library);
            return pipeline;
        }


void BindPCF(ObjCObject encoder,bool enabled)
{
    constexpr NSUInteger ReceiverSlot=6;
    const auto constants=BuildPCFConstants(FrameShadowFilter,enabled);
    Send<void,const void*,NSUInteger,NSUInteger>(encoder,"setFragmentBytes:length:atIndex:",&constants,sizeof(constants),ReceiverSlot);
}
void DestroyPCF()
{
    for (auto& pair:PCFSurfacePipelines) ReleaseObject(pair.second);
    for (auto& pair:PCFScreenPipelines) ReleaseObject(pair.second);
    PCFSurfacePipelines.clear(); PCFScreenPipelines.clear();
    for (auto& library:PCFLibraries) { ReleaseObject(library); library=nullptr; }
}
