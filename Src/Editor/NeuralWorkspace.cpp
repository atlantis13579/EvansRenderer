#include "Editor/NeuralWorkspace.h"
#include <fstream>
#include <thread>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
extern char** environ;
#endif

namespace Evans::Editor
{
namespace
{
#ifdef _WIN32
std::wstring Quote(const std::wstring& text)
{
    std::wstring result=L"\"";size_t slashes=0;
    for(auto ch:text) { if(ch==L'\\') { ++slashes;continue; }result.append(slashes*(ch==L'"'?2:1),L'\\');slashes=0;if(ch==L'"') result+=L'\\';result+=ch; }
    result.append(slashes*2,L'\\');result+=L'"';return result;
}
#endif
void RunTrainer(const std::filesystem::path& executable,const std::filesystem::path& config,const std::filesystem::path& log,
    const std::atomic<bool>& cancelled,const std::function<void(std::string)>& status)
{
    const bool standalone=executable.stem()=="elman_train";
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof(security),nullptr,TRUE};
    HANDLE output=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&security,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(output==INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open training log");
    HANDLE input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&security,OPEN_EXISTING,0,nullptr);
    STARTUPINFOW startup{};startup.cb=sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES;startup.hStdOutput=output;startup.hStdError=output;startup.hStdInput=input;
    auto command=Quote(executable.wstring())+(standalone?L" ":L" train ")+Quote(config.wstring());PROCESS_INFORMATION process{};
    const BOOL started=CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,config.parent_path().c_str(),&startup,&process);
    const auto error=GetLastError();CloseHandle(output);if(input!=INVALID_HANDLE_VALUE) CloseHandle(input);
    if(!started) throw std::runtime_error("Cannot start ElmanLLM: Windows error "+std::to_string(error));CloseHandle(process.hThread);
    unsigned ticks=0;bool stopped=false;
    while(WaitForSingleObject(process.hProcess,100)==WAIT_TIMEOUT) {
        if(cancelled.load()) { TerminateProcess(process.hProcess,1);WaitForSingleObject(process.hProcess,INFINITE);stopped=true;break; }
        if(++ticks%10==0) { std::ifstream stream(log);std::string line,last;while(std::getline(stream,line)) if(!line.empty()) last=line;if(!last.empty()) status(last); }
    }
    DWORD code=1;GetExitCodeProcess(process.hProcess,&code);CloseHandle(process.hProcess);
    if(stopped) throw std::runtime_error("Training cancelled; previously saved assets are unchanged");
#else
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,log.c_str(),O_CREAT|O_WRONLY|O_TRUNC,0644);posix_spawn_file_actions_adddup2(&actions,STDOUT_FILENO,STDERR_FILENO);
    std::string exe=executable.string(),json=config.string();std::vector<char*> args{exe.data()};if(!standalone) args.push_back(const_cast<char*>("train"));args.push_back(json.data());args.push_back(nullptr);
    pid_t child=0;const int launch=posix_spawn(&child,exe.c_str(),&actions,nullptr,args.data(),environ);posix_spawn_file_actions_destroy(&actions);
    if(launch) throw std::runtime_error("Cannot start ElmanLLM");int state=0;
    while(waitpid(child,&state,WNOHANG)==0) { if(cancelled.load()) { kill(child,SIGTERM);waitpid(child,&state,0);throw std::runtime_error("Training cancelled"); }std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
    const int code=WIFEXITED(state)?WEXITSTATUS(state):1;(void)status;
#endif
    if(code!=0) { std::ifstream stream(log);std::string line,last;while(std::getline(stream,line)) if(!line.empty()) last=line;throw std::runtime_error("ElmanLLM failed: "+last+" (see "+log.u8string()+")"); }
}
}
NeuralWorkspace::NeuralWorkspace(std::filesystem::path contentRoot):ContentRoot_(std::move(contentRoot))
{
    const auto root=ContentRoot_.parent_path();Preferences_=root/"Intermediate/Neural.settings.json";
    Options.DataDirectory="Intermediate/Neural/Teapot";Options.OutputDirectory="Contents/Neural/Teapot";
#ifdef _WIN32
    Executable="../ElmanLLM/Binaries/Windows/elman.exe";
#else
    Executable="../ElmanLLM/Binaries/Linux/linux/elman_train";Options.Backend="cpu";
#endif
    Configuration=Options.DataDirectory/"training.json";ModelFile=Options.OutputDirectory/"model.glb";
    try { if(std::filesystem::exists(Preferences_)) {
        const auto json=Neural::Json::read(Preferences_);
        const auto executable=Relative(std::filesystem::u8path(json.at("executable").string()));
        const auto data=Relative(std::filesystem::u8path(json.at("data").string()));
        const auto output=Relative(std::filesystem::u8path(json.at("output").string()));
        if(!executable.is_absolute() && std::filesystem::is_regular_file(Absolute(executable))) Executable=executable;
        if(!data.is_absolute()) Options.DataDirectory=data;
        if(!output.is_absolute()) Options.OutputDirectory=output;
        Configuration=Options.DataDirectory/"training.json";ModelFile=Options.OutputDirectory/"model.glb";
    } } catch(const std::exception& e) { SetMessage(e.what()); }
    if(!HasAsset()) {
        const auto legacy=Options.OutputDirectory/"model.npy";
        std::error_code error;
        if(std::filesystem::is_regular_file(Absolute(legacy),error)) ModelFile=legacy;
    }
}
NeuralWorkspace::~NeuralWorkspace() { Cancel();if(Future_.valid()) Future_.wait(); }
std::filesystem::path NeuralWorkspace::Absolute(const std::filesystem::path& path) const
{ return path.is_absolute()?path.lexically_normal():(ContentRoot_.parent_path()/path).lexically_normal(); }
std::filesystem::path NeuralWorkspace::Relative(const std::filesystem::path& path) const
{ if(!path.is_absolute()) return path.lexically_normal();auto relative=path.lexically_relative(ContentRoot_.parent_path());return relative.empty()?path.lexically_normal():relative; }
bool NeuralWorkspace::HasAsset() const
{ std::error_code error;return std::filesystem::is_regular_file(Absolute(ModelFile),error); }
void NeuralWorkspace::SetMessage(std::string message) { std::lock_guard<std::mutex> lock(Mutex_);Message_=std::move(message); }
std::string NeuralWorkspace::Message() const { std::lock_guard<std::mutex> lock(Mutex_);return Message_; }
void NeuralWorkspace::SavePreferences() const
{
    if(!PersistPreferences) return;
    Neural::Json(Neural::Json::Object{{"executable",Executable.u8string()},{"configuration",Configuration.u8string()},{"model",ModelFile.u8string()},
        {"data",Options.DataDirectory.u8string()},{"output",Options.OutputDirectory.u8string()}}).write(Preferences_.u8string());
}
void NeuralWorkspace::Generate()
{
    if(Busy()) return;Cancelled_=false;Training_=false;auto options=Options;
    options.DataDirectory=Absolute(options.DataDirectory);options.OutputDirectory=Absolute(options.OutputDirectory);
    SetMessage("Generating teapot views and JSON in the data directory...");
    Future_=std::async(std::launch::async,[this,options] { return Neural::GenerateTeapotJob(ContentRoot_,options); });
}
void NeuralWorkspace::StartTraining()
{
    if(Busy()) return;
    TrainAfterGeneration_=true;
    Generate();
}
void NeuralWorkspace::Train()
{
    if(Busy()) return;
    try {
        SavePreferences();const auto configuration=Absolute(Configuration),executable=Absolute(Executable);
        const auto json=Neural::Json::read(configuration);if(json.at("format").string()!="elman.training_graph") throw std::runtime_error("Select an Elman training graph JSON");
        std::filesystem::path model;
        for(const auto& file:json.at("output").at("tensors").array()) {
            const auto destination=std::filesystem::u8path(file.at("file").string());if(destination.extension()==".npy") { model=configuration.parent_path()/destination;break; }
        }
        if(model.empty()) throw std::runtime_error("The JSON must declare a Gaussian NPY output");
        Cancelled_=false;Training_=true;SetMessage("ElmanLLM is reading the selected JSON...");
        const auto asset=Absolute(Options.OutputDirectory/"model.glb");
        Future_=std::async(std::launch::async,[this,configuration,executable,model,asset] {
            RunTrainer(executable,configuration,configuration.parent_path()/"training.log",Cancelled_,[this](auto message){SetMessage(std::move(message));});
            if(Cancelled_.load()) throw std::runtime_error("Training cancelled");
            Neural::SaveGlb(asset,Neural::LoadScene(model));
            return asset;
        });
    } catch(const std::exception& e) { SetMessage(e.what()); }
}
void NeuralWorkspace::Cancel() { Cancelled_=true;TrainAfterGeneration_=false; }
void NeuralWorkspace::Poll()
{
    if(!Future_.valid() || Future_.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready) return;
    try {
        const auto result=Future_.get();if(Cancelled_.load()) { SetMessage("Task cancelled");return; }
        if(Training_) {
            ModelFile=Relative(result);SavePreferences();
            SetMessage("Training finished. Select 3D Gaussian Splatting to load the model.");
            return;
        }
        Configuration=Relative(result);SetMessage("Dataset and JSON ready.");SavePreferences();
        if(TrainAfterGeneration_) { TrainAfterGeneration_=false;Train(); }
    } catch(const std::exception& e) { TrainAfterGeneration_=false;SetMessage(e.what()); }
}
bool NeuralWorkspace::Load()
{
    try { auto model=std::make_shared<Neural::GaussianScene>(Neural::LoadScene(Absolute(ModelFile)));Model=std::move(model);SavePreferences();SetMessage("Loaded "+std::to_string(Model->Points.size())+" Gaussians.");return true; }
    catch(const std::exception& e) { SetMessage(e.what());return false; }
}
}
