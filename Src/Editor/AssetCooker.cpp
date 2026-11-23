#include "Editor/AssetCooker.h"
#include "Editor/ContentRoot.hpp"
#include "Editor/Demos.hpp"
#include "Editor/ModuleAdapters.h"

#include <iostream>

namespace Evans::Editor
{
    std::optional<int> RunAssetCookCommand(const std::vector<std::string>& arguments)
    {
        if (arguments.size()>=2 && arguments[1]=="--bake-lightmap")
        {
            if (arguments.size()<5 || arguments.size()>9)
            { std::cerr<<"Usage: EvansEditor --bake-lightmap ARVO_EDITOR OUTPUT_DIRECTORY triangle|teapot|sponza|dragon [resolution] [samples] [bounces] [cuda|cpu] (Sponza writes PBR lighting to Contents/Scenes/Sponza/Baked; default backend: cuda)\n"; return 2; }
            try
            {
                DemoId demo;
                if(arguments[4]=="triangle") demo=DemoId::Triangle;
                else if(arguments[4]=="teapot") demo=DemoId::UtahTeapot;
                else if(arguments[4]=="sponza") demo=DemoId::SponzaAtrium;
                else if(arguments[4]=="dragon") demo=DemoId::StanfordDragon;
                else throw std::runtime_error("Unknown bake scene");
                DemoCatalog catalog(ResolveContentRoot()); auto scene=catalog.Load(demo);
                Techniques::Settings settings; if(demo==DemoId::SponzaAtrium) DemoCatalog::ApplySponzaAppearance(settings);
                Renderer::ArvoBakeOptions options; options.Executable=std::filesystem::u8path(arguments[2]); options.Directory=std::filesystem::u8path(arguments[3]); options.ProjectRoot=ResolveContentRoot().parent_path();
                if(demo==DemoId::SponzaAtrium)
                {
                    options.Directory=scene.BakedLightingDirectory;
                    settings.Shading=Techniques::ShadingModel::PBR;
                }
                if(arguments.size()>5) settings.GlobalIllumination.Lightmaps.Resolution=std::stoul(arguments[5]);
                if(arguments.size()>6) options.Samples=std::stoul(arguments[6]);
                if(arguments.size()>7) options.Bounces=std::stoul(arguments[7]);
                if(arguments.size()>8)
                {
                    if(arguments[8]!="cuda" && arguments[8]!="cpu") throw std::runtime_error("Expected bake backend cuda|cpu");
                    if(arguments[8]=="cpu") options.Backend=Renderer::ArvoBakeBackend::CPU;
                }
                std::string error;
                if(!BakeSceneWithArvo(scene,settings,options,error,{},[](const auto& message){std::cout<<message<<'\n'<<std::flush;}))
                { std::cerr<<error<<'\n'; return 1; }
                std::cout<<"Lightmap saved to "<<options.Directory.u8string()<<'\n'; return 0;
            }
            catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
        }
        if (arguments.size()<2 || arguments[1]!="--cook-nanite") return std::nullopt;
        // EvansEditor --cook-nanite [all|triangle|teapot|sponza|dragon]
        // This production cooker runs without a window, graphics device or new
        // source assets. Derived archives share the renderer's intermediate cache.
        if (arguments.size()>3)
        { std::cerr<<"Usage: EvansEditor --cook-nanite [all|triangle|teapot|sponza|dragon]\n"; return 2; }
        const auto selection=arguments.size()>2?arguments[2]:"all";
        const std::pair<const char*,DemoId> demos[]={{"triangle",DemoId::Triangle},{"teapot",DemoId::UtahTeapot},
            {"sponza",DemoId::SponzaAtrium},{"dragon",DemoId::StanfordDragon}};
        bool matched=false;
        try
        {
            DemoCatalog catalog(ResolveContentRoot());
            for (const auto& demo:demos)
            {
                if (selection!="all" && selection!=demo.first) continue;
                matched=true; std::size_t families=0; std::string error;
                if (!CookNaniteAssets(catalog.Load(demo.second),families,error))
                { std::cerr<<"Nanite cook failed ("<<demo.first<<"): "<<error<<'\n'; return 1; }
                std::cout<<"Nanite cooked "<<demo.first<<": "<<families<<" families\n"<<std::flush;
            }
        }
        catch (const std::exception& exception) { std::cerr<<"Nanite cook failed: "<<exception.what()<<'\n'; return 1; }
        if (!matched)
        { std::cerr<<"Usage: EvansEditor --cook-nanite [all|triangle|teapot|sponza|dragon]\n"; return 2; }
        return 0;
    }
}
