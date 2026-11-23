#include "Editor/Editor.hpp"
#include "Editor/Imgui.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <ShObjIdl.h>
#endif

namespace Evans::Editor
{
namespace
{
bool Choose(std::filesystem::path& path,const wchar_t* title,bool folder,const wchar_t* filter=L"*.*")
{
#ifdef _WIN32
    const HRESULT initialized=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);IFileOpenDialog* dialog=nullptr;bool selected=false;
    if(SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dialog)))) {
        DWORD flags=0;dialog->GetOptions(&flags);dialog->SetOptions(flags|FOS_FORCEFILESYSTEM|(folder?FOS_PICKFOLDERS:FOS_FILEMUSTEXIST));dialog->SetTitle(title);
        if(!folder) { COMDLG_FILTERSPEC types[]={{L"Supported files",filter}};dialog->SetFileTypes(1,types); }
        const auto directory=folder?path:path.parent_path();IShellItem* initial=nullptr;
        if(SUCCEEDED(SHCreateItemFromParsingName(directory.c_str(),nullptr,IID_PPV_ARGS(&initial)))) { dialog->SetFolder(initial);initial->Release(); }
        if(SUCCEEDED(dialog->Show(GetActiveWindow()))) { IShellItem* item=nullptr;if(SUCCEEDED(dialog->GetResult(&item))) { PWSTR name=nullptr;if(SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH,&name))) { path=name;CoTaskMemFree(name);selected=true; }item->Release(); } }
        dialog->Release();
    }
    if(SUCCEEDED(initialized)) CoUninitialize();return selected;
#else
    (void)path;(void)title;(void)folder;(void)filter;return false;
#endif
}
}
void DrawNeuralPanel(EditorCore& editor)
{
    auto& neural=editor.NeuralRendering();
    auto& options=neural.Options;
    const bool idle=!neural.Busy();
    if(imguiButton("ElmanLLM executable...",idle))
    {
        auto selected=neural.Absolute(neural.Executable);
        if(Choose(selected,L"Select ElmanLLM trainer",false,L"*.exe"))
        { neural.Executable=neural.Relative(selected);neural.SavePreferences(); }
    }
    imguiValue(neural.Absolute(neural.Executable).u8string().c_str());
    if(imguiButton("Data directory...",idle))
    {
        auto selected=neural.Absolute(options.DataDirectory);
        if(Choose(selected,L"Select neural training data directory",true))
        { options.DataDirectory=neural.Relative(selected);neural.Configuration=options.DataDirectory/"training.json";neural.SavePreferences(); }
    }
    imguiValue(neural.Absolute(options.DataDirectory).u8string().c_str());
    if(imguiButton("Output directory...",idle))
    {
        auto selected=neural.Absolute(options.OutputDirectory);
        if(Choose(selected,L"Select trained asset directory",true))
        { options.OutputDirectory=neural.Relative(selected);neural.ModelFile=options.OutputDirectory/"model.glb";neural.SavePreferences(); }
    }
    imguiValue(neural.Absolute(options.OutputDirectory).u8string().c_str());
    if(imguiButton("Start ElmanLLM Training",idle)) neural.StartTraining();
    if(imguiButton("Cancel neural task",!idle)) neural.Cancel();
    const auto message=neural.Message();if(!message.empty()) imguiLabel(message.c_str());
}
}
