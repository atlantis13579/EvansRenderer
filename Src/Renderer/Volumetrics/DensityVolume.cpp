#include "Renderer/Volumetrics/DensityVolume.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace Evans::Renderer
{
    namespace
    {
        constexpr std::array<char,8> DensityMagic{'E','V','D','E','N','S','1','\0'};
        constexpr std::uint32_t MaximumDimension=512;
        constexpr std::uint64_t MaximumVoxels=16ull*1024*1024;
        constexpr unsigned FloatBytes=4,ByteBits=8;
        constexpr std::uint64_t HeaderBytes=DensityMagic.size()+3*FloatBytes+6*FloatBytes;

        std::uint32_t Word(std::istream& file)
        {
            std::array<unsigned char,FloatBytes> bytes{};
            if (!file.read(reinterpret_cast<char*>(bytes.data()),bytes.size())) throw std::runtime_error("Truncated density volume");
            std::uint32_t word=0;
            for (unsigned byte=0;byte<FloatBytes;++byte) word|=std::uint32_t(bytes[byte])<<(byte*ByteBits);
            return word;
        }
        float Float(std::uint32_t word)
        {
            static_assert(sizeof(float)==sizeof(word),"Density assets use IEEE 754 binary32");
            float value; std::memcpy(&value,&word,sizeof(value)); return value;
        }
    }

    bool ValidDensityBinding(const DensityVolumeBinding& binding,const TextureDesc& texture) noexcept
    {
        const float values[]{binding.Grid.X,binding.Grid.Y,binding.Grid.Z,binding.Grid.W};
        for (float value:values) if (!std::isfinite(value) || value<1 || value>MaximumDimension || value!=std::floor(value)) return false;
        const auto nx=unsigned(values[0]),ny=unsigned(values[1]),nz=unsigned(values[2]),columns=unsigned(values[3]);
        return binding.Texture && texture.Format==TextureFormat::RGBA8 && texture.ColorSpace==TextureColorSpace::Linear &&
            texture.Width==int(nx*columns) && texture.Height==int(ny*((nz+columns-1)/columns));
    }

    std::shared_ptr<const DensityVolume> LoadDensityVolume(const std::filesystem::path& path,std::string& error)
    {
        try
        {
            std::ifstream file(path,std::ios::binary|std::ios::ate);
            if (!file) throw std::runtime_error("Cannot open density volume: "+path.string());
            const auto length=file.tellg();
            if (length<static_cast<std::streamoff>(HeaderBytes) || length>static_cast<std::streamoff>(HeaderBytes+MaximumVoxels*FloatBytes))
                throw std::runtime_error("Density volume file size is outside its bounded format");
            file.seekg(0);
            std::array<char,DensityMagic.size()> magic{};
            if (!file.read(magic.data(),magic.size()) || magic!=DensityMagic) throw std::runtime_error("Unsupported density volume format");
            auto result=std::make_shared<DensityVolume>();
            std::uint64_t count=1;
            for (auto& dimension:result->Dimensions)
            {
                dimension=Word(file);
                if (!dimension || dimension>MaximumDimension) throw std::runtime_error("Density volume dimensions must be from 1 to 512");
                count*=dimension;
            }
            if (count>MaximumVoxels || static_cast<std::uint64_t>(length)!=HeaderBytes+count*FloatBytes)
                throw std::runtime_error("Density volume payload does not match its dimensions");
            float bounds[6];
            for (auto& value:bounds)
            {
                value=Float(Word(file));
                if (!std::isfinite(value)) throw std::runtime_error("Density volume bounds must be finite");
            }
            for (unsigned axis=0;axis<3;++axis)
                if (bounds[axis]>=bounds[axis+3]) throw std::runtime_error("Density volume bounds must have positive extent");
            result->BoundsMinimum={bounds[0],bounds[1],bounds[2]};
            result->BoundsMaximum={bounds[3],bounds[4],bounds[5]};
            const auto nx=result->Dimensions[0],ny=result->Dimensions[1],nz=result->Dimensions[2];
            for (result->TileColumns=1;result->TileColumns*result->TileColumns<nz;++result->TileColumns) {}
            result->AtlasRGBA8.resize(std::size_t(result->AtlasWidth())*result->AtlasHeight()*FloatBytes);
            // Store float bits losslessly as linear RGBA8 bytes. Native shaders
            // use integer texel coordinates, decode IEEE 754, then interpolate
            // eight densities. Hardware color filtering and sRGB decoding must
            // never operate on this representation. PBRT's GridDensityRegion
            // uses the same cell-centered trilinear sampling convention.
            for (unsigned z=0;z<nz;++z) for (unsigned y=0;y<ny;++y) for (unsigned x=0;x<nx;++x)
            {
                const auto word=Word(file); const auto density=Float(word);
                if (!std::isfinite(density) || density<0) throw std::runtime_error("Density samples must be finite and nonnegative");
                const auto column=(z%result->TileColumns)*nx+x,row=(z/result->TileColumns)*ny+y;
                auto* pixel=result->AtlasRGBA8.data()+(std::size_t(row)*result->AtlasWidth()+column)*FloatBytes;
                for (unsigned byte=0;byte<FloatBytes;++byte) pixel[byte]=static_cast<std::uint8_t>(word>>(byte*ByteBits));
            }
            error.clear(); return result;
        }
        catch (const std::exception& exception) { error=exception.what(); return {}; }
    }
}
