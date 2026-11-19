#include "Renderer/TechniqueShader.h"
#include "Renderer/ShaderPreparationProgress.h"

#include <filesystem>
#include <fstream>
#include <iterator>

namespace Evans::Renderer
{
    std::size_t HdrShaderIndex(Techniques::HdrPassKind pass) noexcept
    {
        switch (pass)
        {
        case Techniques::HdrPassKind::ColorGrading: return 1;
        case Techniques::HdrPassKind::ToneMapping: return 2;
        case Techniques::HdrPassKind::FlareExtract:
        case Techniques::HdrPassKind::FlareFeatures:
        case Techniques::HdrPassKind::FlareBlurHorizontal:
        case Techniques::HdrPassKind::FlareBlurVertical:
        case Techniques::HdrPassKind::FlareComposite: return 3;
        default: return 0;
        }
    }
    const char* HdrShaderFilename(Techniques::HdrPassKind pass) noexcept
    {
        switch (pass)
        {
        case Techniques::HdrPassKind::BloomExtract:
        case Techniques::HdrPassKind::BloomBlurHorizontal:
        case Techniques::HdrPassKind::BloomBlurVertical:
        case Techniques::HdrPassKind::BloomComposite: return "Bloom";
        case Techniques::HdrPassKind::ColorGrading: return "ColorGrading";
        case Techniques::HdrPassKind::ToneMapping: return "ToneMapping";
        case Techniques::HdrPassKind::FlareExtract:
        case Techniques::HdrPassKind::FlareFeatures:
        case Techniques::HdrPassKind::FlareBlurHorizontal:
        case Techniques::HdrPassKind::FlareBlurVertical:
        case Techniques::HdrPassKind::FlareComposite: return "LensFlare";
        }
        return nullptr;
    }

    const char* PostProcessTechniqueFile(Techniques::PostProcessPassKind pass) noexcept
    {
        switch (pass)
        {
        case Techniques::PostProcessPassKind::FXAA: return "FXAA";
        case Techniques::PostProcessPassKind::Vignette: return "Vignette";
        case Techniques::PostProcessPassKind::FilmGrain: return "FilmGrain";
        case Techniques::PostProcessPassKind::ScreenSpaceDistortion: return "ScreenSpaceDistortion";
        case Techniques::PostProcessPassKind::NPR: return "NPR";
        case Techniques::PostProcessPassKind::Count: break;
        }
        return nullptr;
    }

    const char* ShadingTechniqueFile(Techniques::ShadingModel model, bool areaLights) noexcept
    {
        if (areaLights && model == Techniques::ShadingModel::PBR) return "AreaLights";
        switch (model)
        {
        case Techniques::ShadingModel::Phong: return "Phong";
        case Techniques::ShadingModel::Flat: return "Flat";
        case Techniques::ShadingModel::Gouraud: return "Gouraud";
        case Techniques::ShadingModel::PBR: return "PBR";
        case Techniques::ShadingModel::Toon: return "Toon";
        }
        return nullptr;
    }

    bool LoadTechniqueShaderSource(
        const char* api,
        const char* techniqueFile,
        std::string& source,
        std::string& error)
    {
        // Every API owns its source. This loader does no substitution,
        // cross-compilation, source concatenation, or shader include expansion.
        const std::filesystem::path relative =
            std::filesystem::path(api) / techniqueFile;
#if defined(EVANS_SHADER_SOURCE_DIRECTORY)
        const std::filesystem::path path =
            std::filesystem::path(EVANS_SHADER_SOURCE_DIRECTORY) / relative;
#else
        const std::filesystem::path path =
            std::filesystem::path(__FILE__).parent_path().parent_path() / "Shaders" / relative;
#endif
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
        {
            error = "Cannot read the standalone technique shader: " + path.string();
            return false;
        }
        source.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        if (source.empty())
        {
            error = "The standalone technique shader is empty: " + path.string();
            return false;
        }
        error.clear();
        ShaderPreparationScope::SourceLoaded(api, techniqueFile);
        return true;
    }
}
