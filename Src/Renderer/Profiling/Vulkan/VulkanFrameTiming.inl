// TOP/BOTTOM timestamps bracket this queue's complete frame command buffer,
// including transfers. Read only after its existing completion fence; no extra
// submission or blocking query wait is introduced. Mask the advertised counter
// width before converting ticks with timestampPeriod (nanoseconds per tick).
// https://docs.vulkan.org/spec/latest/chapters/queries.html
VkQueryPool FrameTimeQueries=0;
unsigned FrameTimeBits=0;
bool FrameTimeChecked=false,FrameTimeActive=false;
PFN_vkCreateQueryPool TimeCreatePool=nullptr;
PFN_vkDestroyQueryPool TimeDestroyPool=nullptr;
PFN_vkCmdResetQueryPool TimeReset=nullptr;
PFN_vkCmdWriteTimestamp TimeStamp=nullptr;
PFN_vkGetQueryPoolResults TimeResults=nullptr;
void DestroyFrameTiming()
{
    if (FrameTimeQueries && TimeDestroyPool) TimeDestroyPool(Device,FrameTimeQueries,nullptr);
    FrameTimeQueries=0; FrameTimeChecked=FrameTimeActive=false;
    TimeCreatePool=nullptr; TimeDestroyPool=nullptr; TimeReset=nullptr; TimeStamp=nullptr; TimeResults=nullptr;
}
void BeginFrameTiming()
{
    if (!FrameTimeChecked)
    {
        FrameTimeChecked=true;
        TimeCreatePool=reinterpret_cast<PFN_vkCreateQueryPool>(GetDeviceProcAddress(Device,"vkCreateQueryPool"));
        TimeDestroyPool=reinterpret_cast<PFN_vkDestroyQueryPool>(GetDeviceProcAddress(Device,"vkDestroyQueryPool"));
        TimeReset=reinterpret_cast<PFN_vkCmdResetQueryPool>(GetDeviceProcAddress(Device,"vkCmdResetQueryPool"));
        TimeStamp=reinterpret_cast<PFN_vkCmdWriteTimestamp>(GetDeviceProcAddress(Device,"vkCmdWriteTimestamp"));
        TimeResults=reinterpret_cast<PFN_vkGetQueryPoolResults>(GetDeviceProcAddress(Device,"vkGetQueryPoolResults"));
        if (FrameTimeBits && FrameTimeBits<=64 && std::isfinite(DeviceLimits.timestampPeriod) && DeviceLimits.timestampPeriod>0 &&
            TimeCreatePool && TimeDestroyPool && TimeReset && TimeStamp && TimeResults)
        {
            VkQueryPoolCreateInfo create{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; create.queryType=VK_QUERY_TYPE_TIMESTAMP; create.queryCount=2;
            if (TimeCreatePool(Device,&create,nullptr,&FrameTimeQueries)!=VK_SUCCESS) FrameTimeQueries=0;
        }
    }
    FrameTimeActive=FrameTimeQueries!=0;
    if (!FrameTimeActive) return;
    TimeReset(CommandBuffer,FrameTimeQueries,0,2);
    TimeStamp(CommandBuffer,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,FrameTimeQueries,0);
}
void EndFrameTiming()
{
    if (FrameTimeActive) TimeStamp(CommandBuffer,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,FrameTimeQueries,1);
}
void ReadFrameTiming()
{
    if (!FrameTimeActive) return;
    std::uint64_t ticks[2]{};
    if (TimeResults(Device,FrameTimeQueries,0,2,sizeof(ticks),ticks,sizeof(ticks[0]),VK_QUERY_RESULT_64_BIT)!=VK_SUCCESS) return;
    const auto mask=FrameTimeBits==64?UINT64_MAX:(std::uint64_t{1}<<FrameTimeBits)-1;
    constexpr double NanosecondsToMilliseconds=1.0/1000000;
    const double elapsed=double((ticks[1]-ticks[0])&mask)*double(DeviceLimits.timestampPeriod)*NanosecondsToMilliseconds;
    if (std::isfinite(elapsed) && elapsed>=0)
    { PendingFrame.Statistics.GpuFrameMilliseconds=elapsed; PendingFrame.Statistics.GpuFrameTimingAvailable=true; }
}
