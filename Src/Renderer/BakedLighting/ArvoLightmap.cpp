#include "Renderer/BakedLighting/ArvoLightmap.h"
#include "Renderer/BakedLighting/ArvoExchange.h"
#include "Renderer/BakedLighting/BakedLightingAssets.h"
#include "Renderer/Lighting/SceneLightCollection.h"
#include "Maths/ColorSpace.h"
#include "Renderer/Reflections/EnvironmentRadiance.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>
#include <unordered_map>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
extern char** environ;
#endif

namespace Evans::Renderer
{
namespace
{
ArvoLightmap::Vec3 Convert(Float3 v) { return {v.X,v.Y,v.Z}; }
struct Exporter
{
    ArvoLightmap::Job Job;
    std::unordered_map<const ImageData*,uint32_t> Images;
    uint32_t Texture(const std::shared_ptr<const ImageData>& image)
    {
        if(!image || !image->IsValid()) return ArvoLightmap::Missing;
        const auto found=Images.find(image.get()); if(found!=Images.end()) return found->second;
        const auto id=uint32_t(Job.Textures.size()); Images.emplace(image.get(),id);
        Job.Textures.push_back({image->Width,image->Height,image->PixelsRGBA8}); return id;
    }
    void Geometry(const SceneRayQuery& scene,const LightmapAtlas* atlas,bool flat)
    {
        std::unordered_map<uint32_t,uint32_t> materials;
        for(uint32_t i=0;i<scene.GetTriangleCount();++i)
        {
            if(atlas && atlas->Triangles[i].Chart==InvalidLightmapChart) continue;
            SceneRayTriangle source;
            if(!scene.GetTriangle(i,source)) throw std::runtime_error("Cannot export source triangle");
            auto found=materials.find(source.MeshIndex);
            if(found==materials.end())
            {
                MaterialData material; Float4 color;
                if(!scene.GetTriangleMaterial(i,material,color)) throw std::runtime_error("Cannot export source material");
                ArvoLightmap::Material m;
                m.Base={material.BaseColor.X*color.X,material.BaseColor.Y*color.Y,material.BaseColor.Z*color.Z,material.BaseColor.W*color.W};
                m.Emission={material.EmissiveColor.X,material.EmissiveColor.Y,material.EmissiveColor.Z};
                m.AlphaCutoff=material.AlphaCutoff; m.DiffuseScale=scene.UsesPBR()?0.96f:1.f;
                m.Metallic=material.Metallic.value_or(0); m.Diffuse=Texture(material.DiffuseImage);
                m.Normal=Texture(material.NormalImage); m.Opacity=Texture(material.OpacityImage); m.MetallicMap=Texture(material.MetallicRoughnessImage);
                found=materials.emplace(source.MeshIndex,uint32_t(Job.Materials.size())).first; Job.Materials.push_back(m);
            }
            ArvoLightmap::Triangle triangle; triangle.Material=found->second; triangle.CastShadow=source.CastShadow?1u:0u;
            for(size_t corner=0;corner<3;++corner)
            {
                const auto& v=source.Corners[corner]; auto& target=triangle.Corners[corner];
                target.Position=Convert(v.Position); target.Normal=flat?ArvoLightmap::Vec3{}:Convert(v.Normal); target.UV={v.TexCoord.X,v.TexCoord.Y};
                target.Color={Maths::SRGBToLinear(std::clamp(v.Color.X,0.f,1.f)),Maths::SRGBToLinear(std::clamp(v.Color.Y,0.f,1.f)),Maths::SRGBToLinear(std::clamp(v.Color.Z,0.f,1.f)),v.Color.W};
            }
            if(!atlas) Job.Triangles.push_back(triangle);
            else
            {
                const auto& mapping=atlas->Triangles[i]; ArvoLightmap::Receiver receiver;
                receiver.Surface=triangle; receiver.Chart=mapping.Chart;
                for(size_t c=0;c<3;++c) receiver.UV[c]={mapping.UV[c].X,mapping.UV[c].Y};
                Job.Receivers.push_back(receiver);
            }
        }
    }
};
#ifdef _WIN32
std::wstring Quote(const std::wstring& text)
{
    std::wstring result=L"\""; size_t slashes=0;
    for(auto ch:text) { if(ch==L'\\') { ++slashes; continue; } result.append(slashes*(ch==L'"'?2:1),L'\\'); slashes=0; if(ch==L'"') result+=L'\\'; result+=ch; }
    result.append(slashes*2,L'\\'); result+=L'"'; return result;
}
#endif
bool Run(const std::filesystem::path& executable,const std::filesystem::path& input,const std::filesystem::path& output,
    const std::filesystem::path& log,ArvoBakeBackend backend,const std::function<bool()>& cancelled,std::string& error,
    const std::function<void(const std::string&)>& status)
{
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
    HANDLE file=CreateFileW(log.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&security,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE) { error="Cannot create Arvo log"; return false; }
    STARTUPINFOW startup{}; startup.cb=sizeof(startup); startup.dwFlags=STARTF_USESTDHANDLES; startup.hStdOutput=file; startup.hStdError=file;
    startup.hStdInput=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&security,OPEN_EXISTING,0,nullptr);
    PROCESS_INFORMATION process{};
    auto command=Quote(executable.wstring())+L" --bake-lightmap "+Quote(input.wstring())+L" "+Quote(output.wstring())+(backend==ArvoBakeBackend::CUDA?L" --backend cuda":L" --backend cpu");
    const BOOL started=CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,executable.parent_path().c_str(),&startup,&process);
    const auto launchError=GetLastError(); CloseHandle(file); if(startup.hStdInput!=INVALID_HANDLE_VALUE) CloseHandle(startup.hStdInput);
    if(!started) { error="Cannot launch Arvo (Windows error "+std::to_string(launchError)+")"; return false; }
    CloseHandle(process.hThread); bool stopped=false; unsigned ticks=0;
    while(WaitForSingleObject(process.hProcess,100)==WAIT_TIMEOUT)
    {
        if(cancelled && cancelled()) { TerminateProcess(process.hProcess,1); WaitForSingleObject(process.hProcess,INFINITE); stopped=true; break; }
        if(status && ++ticks%10==0) { std::ifstream stream(log); std::string line,last; while(std::getline(stream,line)) if(!line.empty()) last=line; if(!last.empty()) status("Arvo: "+last); }
    }
    DWORD code=1; GetExitCodeProcess(process.hProcess,&code); CloseHandle(process.hProcess);
#else
    (void)status;
    posix_spawn_file_actions_t actions; posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,log.c_str(),O_CREAT|O_WRONLY|O_TRUNC,0644);
    posix_spawn_file_actions_adddup2(&actions,STDOUT_FILENO,STDERR_FILENO);
    std::string exe=executable.string(),in=input.string(),out=output.string();
    char* arguments[]={exe.data(),const_cast<char*>("--bake-lightmap"),in.data(),out.data(),const_cast<char*>("--backend"),const_cast<char*>(backend==ArvoBakeBackend::CUDA?"cuda":"cpu"),nullptr};
    pid_t child=0; const int launch=posix_spawn(&child,exe.c_str(),&actions,nullptr,arguments,environ);
    posix_spawn_file_actions_destroy(&actions);
    if(launch) { error="Cannot launch Arvo: "+std::to_string(launch); return false; }
    int state=0; bool stopped=false;
    while(waitpid(child,&state,WNOHANG)==0) { if(cancelled && cancelled()) { kill(child,SIGTERM); waitpid(child,&state,0); stopped=true; break; } std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
    const int code=WIFEXITED(state)?WEXITSTATUS(state):1;
#endif
    if(stopped) { error="Arvo lightmap bake cancelled"; return false; }
    if(code) { std::ifstream stream(log); std::string line,last; while(std::getline(stream,line)) if(!line.empty()) last=line; error="Arvo failed: "+last+" (log: "+log.u8string()+")"; return false; }
    return true;
}
}
bool BakeWithArvo(const LightmapBakeRequest& request,const ArvoBakeOptions& options,std::string& error,
    const std::function<bool()>& cancelled,const std::function<void(const std::string&)>& status)
{
    try
    {
        const auto executable=std::filesystem::absolute(options.Executable.is_absolute()?options.Executable:options.ProjectRoot/options.Executable);
        if(!std::filesystem::is_regular_file(executable)) throw std::runtime_error("Select the ArvoEditor executable");
        if(options.Directory.empty()) throw std::runtime_error("Select a lightmap output directory");
        auto stop=[&] { if(cancelled && cancelled()) throw std::runtime_error("Arvo lightmap bake cancelled"); };
        auto report=[&](const char* message) { stop(); if(status) status(message); };
        report("Preparing UV2 and LOD receivers...");
        auto receivers=std::make_shared<LightmapReceivers>();
        if(!BuildLightmapReceivers(request.Instances,request.Materials,request.Atlas,*receivers,error,cancelled)) return false;
        report("Exporting scene to Arvo...");
        Exporter exporter; const bool flat=request.Materials.Shading==Techniques::ShadingModel::Flat;
        exporter.Geometry(*receivers->TransportScene,nullptr,flat); stop();
        exporter.Geometry(*receivers->Atlas->Scene,receivers->Atlas.get(),flat); stop();
        auto& job=exporter.Job; job.Options.Resolution=request.Atlas.Resolution;
        job.Options.Samples=options.Samples; job.Options.Bounces=options.Bounces; job.Options.Threads=options.Threads;
        job.Options.Seed=options.Seed; job.Options.BackFaces=options.BackFaces?1u:0u;
        job.Options.Bias=options.SurfaceBias>0?options.SurfaceBias:request.Sampling.SurfaceBias;
        if(options.Environment && options.EnvironmentScale>0)
        {
            job.EnvironmentSide=options.Environment->Resolution;
            for(uint32_t face=0;face<6;++face) for(uint32_t y=0;y<job.EnvironmentSide;++y) for(uint32_t x=0;x<job.EnvironmentSide;++x)
            {
                const auto value=options.Environment->Texel(EnvironmentFace(face),x,y); const float scale=options.EnvironmentScale;
                job.EnvironmentFaces.push_back({value.X*scale,value.Y*scale,value.Z*scale});
            }
        }
        for(const auto& chart:receivers->Atlas->Charts) job.Charts.push_back({chart.X,chart.Y,chart.Width,chart.Height});
        const auto lights=request.Lighting.SceneLights?request.Lighting.SceneLights->GetLights():
            std::vector<SceneLight>(request.Lighting.Lights.begin(),request.Lighting.Lights.begin()+std::min(request.Lighting.LightCount,MaximumSceneLights));
        for(const auto& light:lights)
        {
            ArvoLightmap::Light l; l.Type=uint32_t(light.Type); l.CastShadow=light.CastShadows?1u:0u;
            l.Position=Convert(light.Position); l.Direction=Convert(light.Direction); l.Color=Convert(light.Color);
            l.Intensity=light.Intensity; l.Range=light.Range; l.Size={std::abs(light.AreaSize.X),std::abs(light.AreaSize.Y)};
            // Legacy rectangle intensity is integrated radiance, whereas prepared lights already carry Le.
            if(l.Type==2 && !light.AreaSampleAxisCount) l.Intensity/=std::max(l.Size.X*l.Size.Y,1e-20f);
            if(l.Type!=2 || (l.Size.X>0 && l.Size.Y>0)) job.Lights.push_back(l);
        }
        const auto directory=std::filesystem::absolute(options.Directory);
        std::filesystem::create_directories(directory);
        const auto workspaceRoot=options.ProjectRoot.empty() ? directory :
            std::filesystem::absolute(options.ProjectRoot/"Intermediate"/"ArvoLightmap");
        std::filesystem::create_directories(workspaceRoot);
        const auto serial=std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const auto workspace=workspaceRoot/("arvo-job-"+std::to_string(serial)); std::filesystem::create_directory(workspace);
        const auto input=workspace/"scene.arvolm",output=workspace/"result.arvolm",log=workspace/"bake.log";
        ArvoLightmap::TransferJob(input,job,false); job={};
        report("Arvo is baking (progress in bake.log)...");
        if(!Run(executable,input,output,log,options.Backend,cancelled,error,status)) return false;
        report("Validating and importing Arvo lightmap...");
        ArvoLightmap::Result result; ArvoLightmap::TransferResult(output,result,true);
        const size_t count=size_t(request.Atlas.Resolution)*request.Atlas.Resolution;
        if(result.Resolution!=request.Atlas.Resolution || result.Front.size()!=count || result.Back.size()!=(options.BackFaces?count:0) || result.Kinds.size()!=count)
            throw std::runtime_error("Arvo returned incompatible lightmap dimensions");
        auto bake=std::make_shared<BakedLightmap>(); bake->Atlas=receivers->Atlas; bake->Settings=request.Sampling;
        bake->Settings.RayCount=options.Samples; bake->Settings.BakeBackFaces=options.BackFaces; bake->Settings.SurfaceBias=options.SurfaceBias>0?options.SurfaceBias:request.Sampling.SurfaceBias;
        auto pixels=[&](const std::vector<ArvoLightmap::Vec4>& source,std::vector<Float4>& target) {
            target.reserve(source.size()); for(const auto& p:source) { if(!std::isfinite(p.X)||!std::isfinite(p.Y)||!std::isfinite(p.Z)||!std::isfinite(p.W)||p.X<0||p.Y<0||p.Z<0||(p.W!=0 && p.W!=1)) throw std::runtime_error("Arvo returned invalid irradiance"); target.push_back({p.X,p.Y,p.Z,p.W}); }
        };
        pixels(result.Front,bake->FrontIrradiance); pixels(result.Back,bake->BackIrradiance);
        for(size_t i=0;i<count;++i) { const auto kind=result.Kinds[i]; if(kind>2 || (kind==0)!=(result.Front[i].W==0)) throw std::runtime_error("Arvo returned invalid coverage"); bake->TexelKinds.push_back(LightmapTexelKind(kind)); bake->CoveredTexelCount+=kind==1; bake->GutterTexelCount+=kind==2; }
        LightmapBakeResult imported; imported.Receivers=receivers; imported.Irradiance=bake;
        stop(); if(!SaveLightmapAsset(directory/"irradiance.evlightmap",request,imported,error)) return false;
        if(!LoadLightmapAsset(directory/"irradiance.evlightmap",request)) throw std::runtime_error("Saved Arvo archive failed runtime reload validation");
        // Keep the log and input as a reproducible standalone Arvo job.
        if(status) status("Arvo lightmap saved"); error.clear(); return true;
    }
    catch(const std::exception& exception) { error=exception.what(); return false; }
}
}
