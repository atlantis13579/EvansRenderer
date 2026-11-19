#pragma once

#include "Renderer/Hair/HairStrands.h"
#include "Renderer/RealtimeShaderContract.h"

namespace Evans::Renderer
{
    inline constexpr unsigned LumenHairMinimumSide=4,LumenHairMaximumSide=64;
    inline constexpr std::size_t LumenHairMaximumCells=std::size_t(LumenHairMaximumSide)*LumenHairMaximumSide*LumenHairMaximumSide;
    struct LumenHairBuildSettings
    {
        unsigned GridSide=32; // Longest axis; cubic world-space cells.
        std::size_t MaximumCells=LumenHairMaximumCells;
    };
    // Low-frequency fiber proxy: rho = sum(2 * radius * centerline length) / V.
    // The normalized second moment C = E[t t^T] retains crossing orientations.
    // A directional extinction approximation is rho * sqrt(max(0,1-w^T C w)).
    // Jensen's inequality makes this an upper estimate of the cell-average
    // projected width, not a conservative geometric cylinder intersection.
    // Kajiya and Kay, "Rendering Fur with Three Dimensional Textures" (1989).
    // PBRT 4e, "Scattering from Hair" and "Transmittance":
    // https://pbr-book.org/4ed/Volume_Scattering/Transmittance
    struct alignas(16) LumenHairVoxel
    {
        Float4 DensityAlbedo{0,0,0,0}; // rho, mean linear RGB reflectance.
        Float4 EmissionCaster{0,0,0,0}; // Mean emitted RGB radiance, caster-only rho.
        Float4 Orientation{0,0,0,0}; // Cxx, Cyy, Czz, Cxy.
        Float4 Cross{0,0,0,0}; // Cxz, Cyz; reserved ZW.
        Float4 CasterOrientation{0,0,0,0}; // Caster-only Cxx, Cyy, Czz, Cxy.
        Float4 CasterCross{0,0,0,0}; // Caster-only Cxz, Cyz; reserved ZW.
    };
    struct alignas(16) LumenHairHeader
    {
        ShaderUInt4 Grid; // XYZ cells; absolute first-voxel word offset when packed.
        Float4 OriginCell{0,0,0,0}; // Minimum world corner, cubic cell side.
    };
    static_assert(sizeof(LumenHairVoxel)==96 && sizeof(LumenHairHeader)==32,"Lumen hair word ABI changed");
    struct LumenHairVolume
    {
        LumenHairHeader Header; // Grid.W remains zero until appended to a native word buffer.
        std::vector<LumenHairVoxel> Voxels;
        unsigned Segments=0,ActiveCells=0;
        double ProjectedArea=0,CasterProjectedArea=0;
    };
    // Consume the existing camera-independent ribbon packet, counting each
    // centerline segment once. Exact cell crossings conserve projected area;
    // two-point quadrature averages varying radius/material/tint per cell.
    // Empty grooms publish an empty volume. Failure/cancellation preserves output.
    bool BuildLumenHairVolume(const HairStrandGeometry&,const Techniques::HairFurSettings&,
        const LumenHairBuildSettings&,LumenHairVolume&,std::string& error,
        const std::function<bool()>& cancelled={});
    bool ValidateLumenHairVolume(const LumenHairVolume&,std::string& error);
    // One aligned header plus typed cells fits the existing single-word-buffer
    // transport binding. Offset zero denotes no hair; failures preserve both outputs.
    bool AppendLumenHairVolume(const LumenHairVolume&,std::size_t maximumWords,
        std::vector<std::uint32_t>& words,unsigned& headerOffset,std::string& error);
    bool ValidateLumenHairWords(const std::vector<std::uint32_t>& words,unsigned headerOffset,
        std::size_t prefixWords,std::string& error);
}
