#include "Editor/AssetCooker.h"
#include "Editor/ContentRoot.hpp"
#include "Editor/Demos.hpp"
#include "Editor/ModuleAdapters.h"

#include <iostream>

namespace Evans::Editor
{
    std::optional<int> RunAssetCookCommand(const std::vector<std::string>& arguments)
    {
        if (arguments.size()>=2 && arguments[1]=="--bake-sponza")
        {
            if (arguments.size()>3 || (arguments.size()==3 && arguments[2]!="pbr")) { std::cerr<<"Usage: EvansEditor --bake-sponza [pbr]\n"; return 2; }
            try
            {
                DemoCatalog catalog(ResolveContentRoot()); auto scene=catalog.Load(DemoId::SponzaAtrium);
                Techniques::Settings settings; DemoCatalog::ApplySponzaAppearance(settings); std::string error;
                if (arguments.size()==3)
                {
                    scene.UsePbrMaterials(true);
                    settings.Shading=Techniques::ShadingModel::PBR;
                }
                if (!CookBakedLightingAssets(scene,settings,error)) { std::cerr<<"Sponza bake failed: "<<error<<'\n'; return 1; }
                std::cout<<"Baked lighting saved to "<<scene.ActiveBakedLightingDirectory().string()<<'\n'<<std::flush; return 0;
            }
            catch (const std::exception& e) { std::cerr<<"Sponza bake failed: "<<e.what()<<'\n'; return 1; }
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
