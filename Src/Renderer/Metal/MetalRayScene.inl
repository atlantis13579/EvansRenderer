// Immutable shared buffers keep BVH publication independent of command-buffer
// submission. The frame owns its snapshot through completion or cancellation.
std::shared_ptr<const RaySceneUpload> FrameRayScene, UploadedRayScene;
std::array<ObjCObject,RaySceneBufferCount> RaySceneBuffers{};
std::array<ObjCObject,4> RayAOPipelines{};

void DestroyRayScene()
{
    DestroyRayRadiance();
    for (auto& buffer:RaySceneBuffers) { ReleaseObject(buffer); buffer=nullptr; }
    for (auto& pipeline:RayAOPipelines) { ReleaseObject(pipeline); pipeline=nullptr; }
    FrameRayScene.reset(); UploadedRayScene.reset();
}

bool EnsureRayScene()
{
    if (!FrameRayScene) return Fail("Metal ray AO requires a scene snapshot");
    if (FrameRayScene==UploadedRayScene) return true;
    if (!ValidateRaySceneUpload(*FrameRayScene,LastError)) return false;
    const auto data=GetRaySceneBufferData(*FrameRayScene);
    const auto selector=Runtime().RegisterSelector("maxBufferLength");
    if (Send<ObjCBool,ObjCSelector>(Device,"respondsToSelector:",selector))
    {
        const auto limit=Send<NSUInteger>(Device,"maxBufferLength");
        for (const auto& buffer:data) if (buffer.Bytes>limit) return Fail("Metal ray scene exceeds maxBufferLength");
    }
    std::array<ObjCObject,RaySceneBufferCount> buffers{};
    for (std::size_t slot=0;slot<data.size();++slot)
    {
        buffers[slot]=Send<ObjCObject,const void*,NSUInteger,NSUInteger>(Device,
            "newBufferWithBytes:length:options:",data[slot].Data,data[slot].Bytes,0);
        if (!buffers[slot])
        {
            for (auto buffer:buffers) ReleaseObject(buffer);
            return Fail("Metal ray-scene buffer allocation failed");
        }
    }
    for (auto buffer:RaySceneBuffers) ReleaseObject(buffer);
    RaySceneBuffers=buffers; UploadedRayScene=FrameRayScene; return true;
}
