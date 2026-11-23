#pragma once
#include "Renderer/Neural/GaussianScene.h"

namespace Evans::Neural
{
struct JobOptions
{
    std::filesystem::path DataDirectory,OutputDirectory;
    int Resolution=96,Views=32,Points=2048,Steps=1200,PixelBatch=256;
    std::string Backend="cuda";
};
std::filesystem::path GenerateTeapotJob(const std::filesystem::path& contentRoot,const JobOptions& options);
void WriteTrainingGraph(const JobOptions& options);
}
