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
bool ChoosePath(std::filesystem::path& path,bool folder)
{
#ifdef _WIN32
    const HRESULT initialized=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    IFileOpenDialog* dialog=nullptr; bool selected=false;
    if(SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dialog))))
    {
        DWORD options=0; dialog->GetOptions(&options); dialog->SetOptions(options|FOS_FORCEFILESYSTEM|(folder?FOS_PICKFOLDERS: FOS_FILEMUSTEXIST));
        dialog->SetTitle(folder?L"Choose lightmap output directory":L"Choose ArvoEditor executable");
        if(!folder) { const COMDLG_FILTERSPEC filters[]={{L"Arvo executable",L"*.exe"}}; dialog->SetFileTypes(1,filters); }
        IShellItem* initial=nullptr;
        const auto directory=folder?path:path.parent_path();
        if(SUCCEEDED(SHCreateItemFromParsingName(directory.c_str(),nullptr,IID_PPV_ARGS(&initial)))) { dialog->SetFolder(initial); initial->Release(); }
        if(SUCCEEDED(dialog->Show(GetActiveWindow())))
        {
            IShellItem* item=nullptr; if(SUCCEEDED(dialog->GetResult(&item))) { PWSTR name=nullptr; if(SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH,&name))) { path=name; CoTaskMemFree(name); selected=true; } item->Release(); }
        }
        dialog->Release();
    }
    if(SUCCEEDED(initialized)) CoUninitialize(); return selected;
#else
    (void)path; (void)folder; return false;
#endif
}
}
void DrawLightmapBakePanel(EditorCore& editor,Techniques::Settings& settings,bool& changed)
{
    static bool expanded=false;
    if(imguiCollapse("Arvo lightmap baking",nullptr,expanded)) expanded=!expanded;
    if(!expanded) return;
    auto& options=editor.LightmapOptions(); const bool idle=!editor.IsLightmapBaking();
    if(imguiButton("Choose ArvoEditor...",idle))
    {
        auto selected=options.Executable.is_absolute()?options.Executable:options.ProjectRoot/options.Executable;
        if(ChoosePath(selected,false))
        {
            options.Executable=selected.lexically_relative(options.ProjectRoot);
            if(options.Executable.empty()) options.Executable=selected;
            editor.SaveLightmapPreferences();
        }
    }
    imguiValue(options.Executable.u8string().c_str());
    const bool fixedOutput=editor.Scene().Demo==DemoId::SponzaAtrium;
    if(imguiButton("Choose output directory...",idle && !fixedOutput) && ChoosePath(options.Directory,true)) editor.SaveLightmapPreferences();
    const auto outputDirectory=editor.LightmapBakeDirectory();
    imguiValue(outputDirectory.u8string().c_str());
    imguiPushEnabled(idle,"A bake is running");
    auto integer=[&](const char* name,uint32_t& value,float minimum,float maximum,float step) {
        float v=float(value); if(imguiSlider(name,&v,minimum,maximum,step)) { value=uint32_t(v); return true; } return false;
    };
    const char* backends[]={"CUDA (offline)","CPU (offline)"};
    int backend=options.Backend==Renderer::ArvoBakeBackend::CUDA?0:1;
    if(imguiCombo("Bake backend",&backend,backends,2))
    { options.Backend=backend==0?Renderer::ArvoBakeBackend::CUDA:Renderer::ArvoBakeBackend::CPU; editor.SaveLightmapPreferences(); }
    auto& atlas=settings.GlobalIllumination.Lightmaps;
    changed|=integer("Atlas resolution",atlas.Resolution,64,4096,64);
    changed|=integer("Chart padding",atlas.Padding,1,16,1);
    changed|=imguiSlider("Chart normal angle",&atlas.MaximumNormalAngleDegrees,1,80,1);
    integer("Samples per texel",options.Samples,1,4096,1);
    integer("Diffuse bounces",options.Bounces,1,16,1);
    if(options.Backend==Renderer::ArvoBakeBackend::CPU) integer("CPU threads (0 = auto)",options.Threads,0,64,1);
    integer("Random seed",options.Seed,1,65535,1);
    imguiSlider("Ray bias (0 = auto)",&options.SurfaceBias,0,0.01f,0.00001f);
    if(imguiCheck("Bake back faces",options.BackFaces)) options.BackFaces=!options.BackFaces;
    imguiPopEnabled();
    imguiLabel("Offline compute; no graphics API. CUDA is the default.");
    imguiLabel("Bakes indirect diffuse light; direct lights stay realtime.");
    if(imguiButton("Bake with Arvo",idle && !editor.IsDemoLoading() && DemoCatalog::SupportsSponzaLightmaps(settings)))
    {
        // Commit edited settings before taking the immutable bake snapshot.
        editor.SetTechniqueSettings(settings); changed=false; editor.StartLightmapBake(editor.Settings().Techniques);
    }
    if(imguiButton("Load baked lightmap",idle)) { editor.LoadLightmapOutput(settings); changed=false; }
    if(imguiButton("Cancel bake",!idle)) editor.CancelLightmapBake();
    const auto message=editor.LightmapBakeMessage(); if(!message.empty()) imguiLabel(message.c_str());
}
}
