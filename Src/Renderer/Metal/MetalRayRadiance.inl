// Immutable scene radiance storage is shared by diffuse GI and specular
// reflections. Each effect retains independent shaders, constants and targets.
RayRadianceFrame FrameRayRadiance;
Techniques::Settings FrameRayRadianceSettings;
std::shared_ptr<const RaySceneShadingUpload> UploadedRayShading;
std::shared_ptr<const SceneLightCollection> UploadedRayLights;
std::array<ObjCObject,RayRadianceBufferCount> RayRadianceBuffers{};

void DestroyRayRadiance()
{
    for (auto& buffer:RayRadianceBuffers) { ReleaseObject(buffer); buffer=nullptr; }
    UploadedRayShading.reset(); UploadedRayLights.reset(); FrameRayRadiance={};
}

bool EnsureRayRadiance()
{
    const auto& frame=FrameRayRadiance;
    if (!frame.Geometry || frame.Geometry!=FrameRayScene || !frame.Shading || !frame.Lights)
        return Fail("Metal ray radiance requires matched geometry, materials and lights");
    if (!EnsureRayScene()) return false;
    const bool materials=frame.Shading!=UploadedRayShading,lights=frame.Lights!=UploadedRayLights;
    if (!materials && !lights) return true;
    if (materials && !ValidateRaySceneShadingUpload(*frame.Geometry,*frame.Shading,LastError)) return false;
    const auto data=GetRayRadianceBufferData(frame);
    const auto selector=Runtime().RegisterSelector("maxBufferLength");
    if (Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",selector))
    {
        const auto limit=Send<NSUInteger>(Device,"maxBufferLength");
        for (const auto& buffer:data) if (buffer.Bytes>limit) return Fail("Metal ray shading exceeds maxBufferLength");
    }
    std::array<ObjCObject,RayRadianceBufferCount> replacements{};
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        if (!(slot<RaySceneShadingBufferCount ? materials : lights)) continue;
        replacements[slot]=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,
            "newBufferWithBytes:length:options:",data[slot].Data,data[slot].Bytes,0);
        if (!replacements[slot])
        {
            for (auto buffer:replacements) ReleaseObject(buffer);
            return Fail("Metal ray radiance buffer allocation failed");
        }
    }
    for (std::size_t slot=0;slot<replacements.size();++slot) if (replacements[slot])
    { ReleaseObject(RayRadianceBuffers[slot]); RayRadianceBuffers[slot]=replacements[slot]; }
    UploadedRayShading=frame.Shading; UploadedRayLights=frame.Lights; return true;
}
