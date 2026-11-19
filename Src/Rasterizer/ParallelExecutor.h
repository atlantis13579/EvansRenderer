#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace Evans::Rasterizer::Detail
{
// A render context reuses its workers across preparation and tile passes.
// Jobs own disjoint output ranges; the submitting thread also executes jobs.
class ParallelExecutor
{
public:
    ParallelExecutor() = default;
    ~ParallelExecutor();
    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;
    void Run(std::uint32_t jobs, std::uint32_t workers,
        const std::function<void(std::uint32_t)>& function);

private:
    void Worker(std::uint32_t index);
    void Execute(const std::function<void(std::uint32_t)>& function, std::uint32_t jobs);
    std::mutex SubmitMutex_, StateMutex_;
    std::condition_variable Ready_, Finished_;
    std::vector<std::thread> Threads_;
    std::function<void(std::uint32_t)> Function_;
    std::atomic<std::uint32_t> Next_{0};
    std::atomic<bool> Failed_{false};
    std::uint64_t Generation_ = 0;
    std::uint32_t Jobs_ = 0, Active_ = 0, Remaining_ = 0;
    bool Stop_ = false;
    std::exception_ptr Error_;
};
}
