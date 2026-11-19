#pragma once
#include "Renderer/Types.h"
#include <array>
#include <cstddef>

namespace Evans::Renderer
{
    enum class MaterialTechnique : unsigned { Basic,Texture,Normal,AlphaTest,AlphaBlend,Count };
    inline constexpr unsigned MaterialTechniqueCount=unsigned(MaterialTechnique::Count);
    inline constexpr unsigned MaterialSurfaceModels=5,MaterialGBufferProgram=MaterialSurfaceModels*2;
    inline constexpr unsigned MaterialProgramsPerTechnique=MaterialGBufferProgram+1;
    inline constexpr unsigned MaterialProgramCount=MaterialTechniqueCount*MaterialProgramsPerTechnique;
    inline constexpr unsigned MaterialPipelineKeyShift=46;
    static_assert(MaterialTechniqueCount<8,"Extend the material pipeline key field");
    // These features compose. The dominant coverage/mapping operation chooses
    // an independently owned stage; all lower material operations remain active.
    inline MaterialTechnique SelectMaterialTechnique(const PipelineState& material,bool weighted=false) noexcept
    {
        if (material.BlendEnabled || weighted) return MaterialTechnique::AlphaBlend;
        if (material.AlphaCutoff>0) return MaterialTechnique::AlphaTest;
        if (material.NormalTexture) return MaterialTechnique::Normal;
        if (material.DiffuseTexture || material.OpacityTexture) return MaterialTechnique::Texture;
        return MaterialTechnique::Basic;
    }
    inline const char* MaterialTechniqueFile(MaterialTechnique technique) noexcept
    {
        constexpr std::array<const char*,MaterialTechniqueCount> names{
            "BasicMaterials","TextureMapping","NormalMapping","AlphaTesting","AlphaBlending"};
        const auto index=unsigned(technique); return index<names.size()?names[index]:nullptr;
    }
    inline unsigned MaterialProgramIndex(MaterialTechnique technique,unsigned model,bool weighted,bool geometry) noexcept
    { return unsigned(technique)*MaterialProgramsPerTechnique+(geometry?MaterialGBufferProgram:model+(weighted?MaterialSurfaceModels:0)); }
}
