#pragma once
#include "Renderer/Neural/TrainingJob.h"
#include <atomic>
#include <future>
#include <mutex>

namespace Evans::Editor
{
class NeuralWorkspace
{
public:
    explicit NeuralWorkspace(std::filesystem::path contentRoot);
    ~NeuralWorkspace();
    Neural::JobOptions Options;
    std::filesystem::path Executable,Configuration,ModelFile;
    std::shared_ptr<const Neural::GaussianScene> Model;
    bool PersistPreferences=true;
    bool Busy() const { return Future_.valid(); }
    std::string Message() const;
    void Generate();
    void StartTraining();
    void Train();
    void Cancel();
    void Poll();
    bool Load();
    bool HasAsset() const;
    std::filesystem::path Absolute(const std::filesystem::path& path) const;
    std::filesystem::path Relative(const std::filesystem::path& path) const;
    void SavePreferences() const;
private:
    std::filesystem::path ContentRoot_,Preferences_;
    std::future<std::filesystem::path> Future_;
    std::atomic<bool> Cancelled_{false};
    bool Training_=false;
    bool TrainAfterGeneration_=false;
    mutable std::mutex Mutex_;
    std::string Message_;
    void SetMessage(std::string message);
};
}
