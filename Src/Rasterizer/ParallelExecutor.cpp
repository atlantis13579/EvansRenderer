#include "Rasterizer/ParallelExecutor.h"
#include <algorithm>

namespace Evans::Rasterizer::Detail
{
ParallelExecutor::~ParallelExecutor()
{
    { std::lock_guard<std::mutex> lock(StateMutex_); Stop_=true; }
    Ready_.notify_all();
    for(auto& thread:Threads_)thread.join();
}
void ParallelExecutor::Execute(const std::function<void(std::uint32_t)>& function,std::uint32_t jobs)
{
    try
    {
        while(!Failed_.load(std::memory_order_relaxed))
        {
            const auto begin=Next_.fetch_add(1,std::memory_order_relaxed);
            if(begin>=jobs)break;
            function(begin);
        }
    }
    catch(...)
    {
        std::lock_guard<std::mutex> lock(StateMutex_);
        if(!Error_)Error_=std::current_exception();
        Failed_.store(true,std::memory_order_relaxed);
    }
}
void ParallelExecutor::Worker(std::uint32_t index)
{
    std::uint64_t generation=0;
    std::unique_lock<std::mutex> lock(StateMutex_);
    for(;;)
    {
        Ready_.wait(lock,[&]{return Stop_ || Generation_!=generation;});
        if(Stop_)return;
        generation=Generation_;
        if(index>=Active_)continue;
        const auto function=Function_;const auto jobs=Jobs_;
        lock.unlock();Execute(function,jobs);lock.lock();
        if(--Remaining_==0)Finished_.notify_one();
    }
}
void ParallelExecutor::Run(std::uint32_t jobs,std::uint32_t workers,
    const std::function<void(std::uint32_t)>& function)
{
    if(!jobs)return;
    std::lock_guard<std::mutex> submission(SubmitMutex_);
    workers=std::max(1u,std::min(jobs,workers?workers:std::max(1u,std::thread::hardware_concurrency())));
    if(workers==1)
    {
        for(std::uint32_t job=0;job<jobs;++job)function(job);
        return;
    }
    while(Threads_.size()<workers-1)
    {
        const auto index=static_cast<std::uint32_t>(Threads_.size());
        Threads_.emplace_back([this,index]{Worker(index);});
    }
    {
        std::lock_guard<std::mutex> lock(StateMutex_);
        Function_=function;Jobs_=jobs;Active_=Remaining_=workers-1;
        Error_=nullptr;Failed_.store(false,std::memory_order_relaxed);Next_.store(0,std::memory_order_relaxed);
        ++Generation_;
    }
    Ready_.notify_all();Execute(function,jobs);
    std::unique_lock<std::mutex> lock(StateMutex_);
    Finished_.wait(lock,[&]{return Remaining_==0;});
    Function_={};
    if(Error_)std::rethrow_exception(Error_);
}
}
