#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace Evans::Renderer
{
    struct ShaderPreparationStatus
    {
        std::string Source;
        std::uint64_t SourcesLoaded = 0;
    };

    // Lazy shader variants have no known total. Report actual source loads;
    // callers must not present this count as a compilation percentage.
    class ShaderPreparationProgress
    {
    public:
        [[nodiscard]] ShaderPreparationStatus Read() const
        {
            std::lock_guard<std::mutex> lock(Mutex);
            return Status;
        }
        void SourceLoaded(const char* api, const char* file)
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Status.Source = std::string(api) + "/" + file;
            ++Status.SourcesLoaded;
        }
    private:
        mutable std::mutex Mutex;
        ShaderPreparationStatus Status;
    };

    // A job owns its observer even when retired. Thread-local binding prevents
    // concurrent backends from publishing into another viewport generation.
    class ShaderPreparationScope
    {
    public:
        explicit ShaderPreparationScope(std::shared_ptr<ShaderPreparationProgress> progress)
            : Previous(std::move(Current)) { Current = std::move(progress); }
        ~ShaderPreparationScope() { Current = std::move(Previous); }
        ShaderPreparationScope(const ShaderPreparationScope&) = delete;
        ShaderPreparationScope& operator=(const ShaderPreparationScope&) = delete;
        static void SourceLoaded(const char* api, const char* file)
        { if (Current) Current->SourceLoaded(api, file); }
        [[nodiscard]] static std::shared_ptr<ShaderPreparationProgress> Capture() { return Current; }
    private:
        inline static thread_local std::shared_ptr<ShaderPreparationProgress> Current;
        std::shared_ptr<ShaderPreparationProgress> Previous;
    };
}
