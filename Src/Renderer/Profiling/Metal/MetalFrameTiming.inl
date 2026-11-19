// Metal reports the completed command buffer's native GPU execution interval,
// including blits. Read before releasing the buffer, after waitUntilCompleted;
// unavailable/zero timestamps never become a synthetic CPU measurement.
// https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime
void ReadFrameTiming()
{
    const auto start=Runtime().RegisterSelector("GPUStartTime"),end=Runtime().RegisterSelector("GPUEndTime");
    if (!Send<ObjCBool,ObjCSelector>(CommandBuffer,"respondsToSelector:",start) ||
        !Send<ObjCBool,ObjCSelector>(CommandBuffer,"respondsToSelector:",end)) return;
    const double begin=Send<double>(CommandBuffer,"GPUStartTime"),finish=Send<double>(CommandBuffer,"GPUEndTime");
    constexpr double MillisecondsPerSecond=1000;
    const double elapsed=(finish-begin)*MillisecondsPerSecond;
    if (std::isfinite(begin) && begin>0 && finish>=begin && std::isfinite(elapsed))
    { PendingFrame.Statistics.GpuFrameMilliseconds=elapsed; PendingFrame.Statistics.GpuFrameTimingAvailable=true; }
}
