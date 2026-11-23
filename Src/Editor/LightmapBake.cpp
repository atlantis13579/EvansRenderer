#include "Editor/Editor.hpp"
#include "Editor/ModuleAdapters.h"
#include <fstream>
#include <iomanip>

namespace Evans::Editor
{
void EditorCore::LoadLightmapPreferences(const std::filesystem::path& contentRoot)
{
    const auto root=contentRoot.parent_path();
    LightmapPreferences_=root/"Intermediate"/"ArvoLightmap.settings";
#ifdef _WIN32
    LightmapOptions_.Executable=std::filesystem::path("..")/"ArvoRenderer"/"Binaries"/"Release"/"ArvoEditor.exe";
#else
    LightmapOptions_.Executable=std::filesystem::path("..")/"ArvoRenderer"/"Binaries"/"Release"/"ArvoEditor";
#endif
    LightmapOptions_.ProjectRoot=root;
    LightmapOptions_.Directory=root/"Output"/"Lightmaps";
    std::ifstream input(LightmapPreferences_); std::string executable,directory;
    auto options=LightmapOptions_;
    if(input>>std::quoted(executable)>>std::quoted(directory)>>options.Samples>>options.Bounces>>options.Threads>>options.Seed>>options.BackFaces>>options.SurfaceBias)
    {
        std::string backend; if(input>>backend) options.Backend=backend=="cpu"?Renderer::ArvoBakeBackend::CPU:Renderer::ArvoBakeBackend::CUDA;
        options.Executable=std::filesystem::u8path(executable); options.Directory=std::filesystem::u8path(directory);
        if(options.Executable.filename()=="ArvoLightmap.exe") options.Executable=std::filesystem::path("..")/"ArvoRenderer"/"Binaries"/"Release"/"ArvoEditor.exe";
        if(options.Samples>=1 && options.Samples<=65536 && options.Bounces<=32 && options.Threads<=256 && options.SurfaceBias>=0)
            LightmapOptions_=options;
    }
}
void EditorCore::SaveLightmapPreferences() const
{
    std::error_code ignored; std::filesystem::create_directories(LightmapPreferences_.parent_path(),ignored);
    std::ofstream out(LightmapPreferences_); const auto& s=LightmapOptions_;
    out<<std::quoted(s.Executable.u8string())<<'\n'<<std::quoted(s.Directory.u8string())<<'\n'
       <<s.Samples<<' '<<s.Bounces<<' '<<s.Threads<<' '<<s.Seed<<' '<<s.BackFaces<<' '<<s.SurfaceBias<<'\n'<<(s.Backend==Renderer::ArvoBakeBackend::CUDA?"cuda":"cpu")<<'\n';
}
std::filesystem::path EditorCore::LightmapBakeDirectory() const
{
    return Scene_.Demo==DemoId::SponzaAtrium ? Scene_.BakedLightingDirectory : LightmapOptions_.Directory;
}
std::string EditorCore::LightmapBakeMessage() const
{
    if(!LightmapTask_) return LightmapMessage_;
    std::lock_guard<std::mutex> lock(LightmapTask_->Mutex); return LightmapTask_->Message;
}
void EditorCore::CancelLightmapBake()
{
    if(LightmapTask_) LightmapTask_->Cancelled.store(true);
}
void EditorCore::StartLightmapBake(const Techniques::Settings& settings)
{
    if(LightmapTask_ || IsDemoLoading()) return;
    SaveLightmapPreferences();
    if(Scene_.Demo==DemoId::UtahTeapot) SetTeapotRotationSpeed(0);
    auto task=std::make_unique<LightmapTask>(); task->Demo=Scene_.Demo;
    task->Revision=Scene_.Revision; task->TransformRevision=Scene_.TransformRevision; task->LightingRevision=Scene_.LightingRevision;
    task->Settings=settings; task->Directory=LightmapBakeDirectory(); task->Atlas=settings.GlobalIllumination.Lightmaps; task->Message="Preparing Arvo bake...";
    auto* progress=task.get(); auto options=LightmapOptions_; options.Directory=task->Directory; const auto scene=Scene_;
    auto bakeSettings=settings;
    if (scene.Demo==DemoId::SponzaAtrium) bakeSettings.Shading=Techniques::ShadingModel::PBR;
    try
    {
        task->Future=std::async(std::launch::async,[scene,bakeSettings,options,progress] {
            std::string error;
            try
            {
                if(!BakeSceneWithArvo(scene,bakeSettings,options,error,[progress]{return progress->Cancelled.load();},[progress](const auto& message) {
                    std::lock_guard<std::mutex> lock(progress->Mutex); progress->Message=message;
                })) return error;
                return std::string();
            }
            catch(const std::exception& e) { return std::string(e.what()); }
        });
        LightmapTask_=std::move(task);
    }
    catch(const std::exception& e) { LightmapMessage_=e.what(); }
}
void EditorCore::LoadLightmapOutput(const Techniques::Settings& settings)
{
    if(LightmapTask_) return;
    SetTechniqueSettings(settings);
    Scene_.LightmapDirectory=LightmapBakeDirectory();
    Settings_.Techniques.GlobalIllumination.Mode=Techniques::GlobalIlluminationMode::Lightmaps;
    ++Scene_.Revision; ++RenderHistoryEpoch_;
    LightmapMessage_="Loading lightmap; geometry and atlas settings must match.";
    SaveLightmapPreferences(); InvalidateAndRender();
}
void EditorCore::PollLightmapBake()
{
    if(!LightmapTask_ || LightmapTask_->Future.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready) return;
    auto task=std::move(LightmapTask_); const auto error=task->Future.get();
    if(!error.empty()) { LightmapMessage_=error; return; }
    if(task->Demo!=Scene_.Demo || task->Revision!=Scene_.Revision || task->TransformRevision!=Scene_.TransformRevision || task->LightingRevision!=Scene_.LightingRevision || task->Settings!=Settings_.Techniques)
    { LightmapMessage_="Bake saved. Scene changed; result was not applied."; return; }
    Scene_.LightmapDirectory=task->Directory;
    Settings_.Techniques.GlobalIllumination.Lightmaps=task->Atlas;
    Settings_.Techniques.GlobalIllumination.Mode=Techniques::GlobalIlluminationMode::Lightmaps;
    ++Scene_.Revision; ++RenderHistoryEpoch_; LightmapMessage_="Arvo lightmap saved; loading result...";
    InvalidateAndRender();
}
}
