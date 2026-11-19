#include "Renderer/RayTracing/SceneRayShadingUpload.h"
#include "Renderer/RayTracing/SceneRayQuery.h"
#include "Maths/ColorSpace.h"

#include <cmath>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float4 value)
        { return std::isfinite(value.X)&&std::isfinite(value.Y)&&std::isfinite(value.Z)&&std::isfinite(value.W); }
        Float4 Attribute(Float3 value) { return {value.X,value.Y,value.Z,0}; }
        bool UnitTexel(Float4 value)
        {
            return Finite(value) && value.X>=0 && value.X<=1 && value.Y>=0 && value.Y<=1 &&
                value.Z>=0 && value.Z<=1 && value.W>=0 && value.W<=1;
        }
    }

    bool ValidateRaySceneShadingUpload(const RaySceneUpload& visibility,const RaySceneShadingUpload& shading,std::string& error)
    {
        if (!ValidateRaySceneUpload(visibility,error)) return false;
        const auto fail=[&](const char* message) { error=message; return false; };
        if (shading.Surfaces.size()!=visibility.Triangles.size() || shading.Materials.size()!=visibility.Materials.size())
            return fail("Ray shading does not match visibility triangle/material counts");
        if (shading.Texels.size()>UINT32_MAX || (shading.Flags&~RaySceneFlatNormals))
            return fail("Ray shading has invalid flags or exceeds 32-bit texel addressing");
        for (const auto& surface:shading.Surfaces)
        {
            for (const auto normal:{surface.NormalA,surface.NormalB,surface.NormalC})
                if (!Finite(normal) || normal.W!=0) return fail("Ray surface has an invalid transformed normal");
            for (const auto color:{surface.ColorA,surface.ColorB,surface.ColorC})
                if (!UnitTexel(color) || color.W!=0) return fail("Ray surface has an invalid linear vertex color");
        }
        for (const auto& material:shading.Materials)
        {
            if (!Finite(material.BaseColor) || !Finite(material.AmbientColor) || !Finite(material.SpecularShininess) ||
                !Finite(material.Emission) || material.SpecularShininess.W<1 || material.AmbientColor.W!=0 || (material.Emission.W!=0 && material.Emission.W<1))
                return fail("Ray surface has invalid material coefficients");
            for (const auto value:{material.SpecularShininess,material.Emission})
                if (value.X<0 || value.Y<0 || value.Z<0) return fail("Ray surface has negative specular or emitted radiance");
            for (const auto& image:{material.DiffuseImage,material.NormalImage})
            {
                if (!image[3])
                { if (image[0]||image[1]||image[2]) return fail("Absent ray shading image has a nonempty range"); }
                else if (image[3]!=RaySceneImagePresent || !image[1] || !image[2] ||
                    std::uint64_t(image[0])+std::uint64_t(image[1])*image[2]>shading.Texels.size())
                    return fail("Ray shading image exceeds its Float4 atlas");
            }
            if ((shading.Flags&RaySceneFlatNormals) && material.NormalImage[3])
                return fail("Flat ray shading must not apply a tangent normal map");
        }
        for (const auto texel:shading.Texels)
            if (!UnitTexel(texel)) return fail("Ray shading texel is not finite UNORM/linear RGB");
        return true;
    }

    bool SceneRayQuery::ExportRadiance(RaySceneUpload& visibility,RaySceneShadingUpload& shading,std::string& error) const
    {
        RaySceneUpload visibilityCandidate;
        if (!Export(visibilityCandidate,error)) return false;
        RaySceneShadingUpload candidate;
        candidate.Flags=FlatShading ? RaySceneFlatNormals : 0;
        candidate.Surfaces.reserve(visibilityCandidate.Triangles.size()); candidate.Materials.reserve(Meshes.size());
        // The same source can be both diffuse and normal data. Encoding belongs
        // in the deduplication key: sRGB decoding must precede interpolation,
        // while tangent normals remain linear even with diffuse mapping off.
        using ImageDescriptor=std::array<std::uint32_t,4>;
        std::unordered_map<const ImageData*,std::array<ImageDescriptor,2>> images;
        auto packImage=[&](const std::shared_ptr<const ImageData>& source,bool srgb,ImageDescriptor& output) {
            output={}; if (!source || !source->IsValid()) return true;
            auto& cached=images[source.get()][srgb ? 1 : 0];
            if (cached[3]) { output=cached; return true; }
            const std::uint64_t count=std::uint64_t(source->Width)*source->Height;
            if (count>UINT32_MAX-candidate.Texels.size()) { error="Ray shading atlas exceeds 32-bit addressing"; return false; }
            output={static_cast<std::uint32_t>(candidate.Texels.size()),source->Width,source->Height,RaySceneImagePresent};
            constexpr float ByteMaximum=255;
            constexpr std::size_t PixelChannels=4;
            for (std::size_t index=0;index<count;++index)
            {
                const auto* bytes=source->PixelsRGBA8.data()+index*PixelChannels;
                Float4 texel {bytes[0]/ByteMaximum,bytes[1]/ByteMaximum,bytes[2]/ByteMaximum,bytes[3]/ByteMaximum};
                if (srgb)
                { texel.X=Maths::SRGBToLinear(texel.X); texel.Y=Maths::SRGBToLinear(texel.Y); texel.Z=Maths::SRGBToLinear(texel.Z); }
                candidate.Texels.push_back(texel);
            }
            cached=output; return true;
        };
        for (const auto& mesh:Meshes)
        {
            const auto& source=mesh.Material;
            if (!Finite(source.AmbientColor) || !Finite(source.SpecularColor) || !std::isfinite(source.Shininess))
            { error="Ray shading requires finite ambient/specular coefficients and shininess"; return false; }
            RaySceneSurfaceMaterial material;
            material.BaseColor={mesh.Color.X*source.BaseColor.X,mesh.Color.Y*source.BaseColor.Y,
                mesh.Color.Z*source.BaseColor.Z,mesh.Color.W*source.BaseColor.W};
            material.AmbientColor={source.AmbientColor.X,source.AmbientColor.Y,source.AmbientColor.Z,0};
            material.SpecularShininess={std::max(source.SpecularColor.X,0.0f),std::max(source.SpecularColor.Y,0.0f),
                std::max(source.SpecularColor.Z,0.0f),std::max(source.Shininess,1.0f)};
            material.Emission={std::max(source.EmissiveColor.X,0.0f),std::max(source.EmissiveColor.Y,0.0f),std::max(source.EmissiveColor.Z,0.0f),source.Refractive?(1+std::sqrt(std::clamp(source.DielectricReflectance,0.000001f,0.99f)))/(1-std::sqrt(std::clamp(source.DielectricReflectance,0.000001f,0.99f))):0};
            if ((TextureMapping && !packImage(source.DiffuseImage,true,material.DiffuseImage)) ||
                (NormalMapping && !packImage(source.NormalImage,false,material.NormalImage))) return false;
            candidate.Materials.push_back(material);
        }
        const auto& indices=Geometry.GetIndices();
        for (const auto& primitive:visibilityCandidate.Triangles)
        {
            const auto first=std::size_t(primitive.Metadata[1])*3;
            const auto a=indices[first],b=indices[first+1],c=indices[first+2];
            candidate.Surfaces.push_back({Attribute(Vertices[a].Normal),Attribute(Vertices[b].Normal),Attribute(Vertices[c].Normal),
                Attribute(LinearVertexColors[a]),Attribute(LinearVertexColors[b]),Attribute(LinearVertexColors[c])});
        }
        if (!ValidateRaySceneShadingUpload(visibilityCandidate,candidate,error)) return false;
        visibility=std::move(visibilityCandidate); shading=std::move(candidate); return true;
    }
}
