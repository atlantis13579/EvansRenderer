#include "Renderer/RayTracing/SceneRayUpload.h"
#include "Renderer/RayTracing/SceneRayQuery.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Evans::Renderer
{
    namespace
    {
        bool Finite(Float4 value)
        { return std::isfinite(value.X)&&std::isfinite(value.Y)&&std::isfinite(value.Z)&&std::isfinite(value.W); }
        bool Range(std::uint32_t first,std::uint64_t count,std::size_t capacity)
        { return first<=capacity && count<=capacity-first; }
        Float4 PositionOpacity(const Vertex& vertex)
        { return {vertex.Position.X,vertex.Position.Y,vertex.Position.Z,vertex.Color.W}; }
        bool Contains(const RaySceneNode& node,Float4 position)
        {
            return position.X>=node.Minimum.X&&position.X<=node.Maximum.X&&
                position.Y>=node.Minimum.Y&&position.Y<=node.Maximum.Y&&
                position.Z>=node.Minimum.Z&&position.Z<=node.Maximum.Z;
        }
    }

    bool ValidateRaySceneGeometry(const std::vector<RaySceneTriangle>& triangles,const std::vector<RaySceneMaterial>& materials,
        const std::vector<float>& coverageTexels,std::string& error)
    {
        error.clear();
        const auto fail=[&](const char* message) { error=message; return false; };
        if (triangles.size()>UINT32_MAX || materials.size()>UINT32_MAX || coverageTexels.size()>UINT32_MAX)
            return fail("Ray geometry exceeds 32-bit shader addressing");
        for (const auto& triangle:triangles)
            if (!Finite(triangle.A)||!Finite(triangle.B)||!Finite(triangle.C)||!Finite(triangle.UVAB)||!Finite(triangle.UVC)||
                triangle.Metadata[0]>=materials.size() || (triangle.Metadata[3]&~RaySceneShadowCaster))
                return fail("Ray triangle has invalid attributes or material");
        for (const auto& material:materials)
        {
            if (!Finite(material.Coverage) || material.Coverage.Y<0)
                return fail("Ray material has invalid coverage");
            for (const auto& image:{material.DiffuseAlpha,material.OpacityRed})
            {
                if (!image[3]) { if (image[0]||image[1]||image[2]) return fail("Absent ray image has a nonempty range"); }
                else if (image[3]!=RaySceneImagePresent || !image[1] || !image[2] ||
                    !Range(image[0],std::uint64_t(image[1])*image[2],coverageTexels.size()))
                    return fail("Ray image exceeds its scalar atlas");
            }
        }
        for (float value:coverageTexels)
            if (!std::isfinite(value)||value<0||value>1) return fail("Ray coverage texel is not in the unit interval");
        return true;
    }

    bool ValidateRaySceneUpload(const RaySceneUpload& scene,std::string& error)
    {
        error.clear();
        auto fail=[&](const char* message) { error=message; return false; };
        if (scene.Nodes.size()>UINT32_MAX || scene.Triangles.size()>UINT32_MAX ||
            scene.Materials.size()>UINT32_MAX || scene.CoverageTexels.size()>UINT32_MAX)
            return fail("Ray scene exceeds 32-bit shader addressing");
        if (scene.Nodes.empty()!=scene.Triangles.empty()) return fail("Ray scene geometry and root disagree");
        if (!scene.Nodes.empty() && scene.Nodes[0].Links[3]!=scene.Nodes.size()) return fail("Ray root has an invalid escape link");
        std::size_t nextTriangle=0;
        for (std::size_t index=0;index<scene.Nodes.size();++index)
        {
            const auto& node=scene.Nodes[index]; const auto& link=node.Links;
            if (!Finite(node.Minimum)||!Finite(node.Maximum)||node.Minimum.X>node.Maximum.X||
                node.Minimum.Y>node.Maximum.Y||node.Minimum.Z>node.Maximum.Z||link[3]<=index||link[3]>scene.Nodes.size())
                return fail("Ray node has invalid bounds or a non-forward escape");
            if (link[2])
            {
                if (link[0]!=RaySceneInvalidIndex || link[3]!=index+1 || link[1]!=nextTriangle ||
                    !Range(link[1],link[2],scene.Triangles.size()))
                    return fail("Ray leaf has an invalid triangle range");
                nextTriangle+=link[2];
                for (std::size_t triangle=link[1];triangle<nextTriangle;++triangle)
                {
                    const auto& primitive=scene.Triangles[triangle];
                    if (!Contains(node,primitive.A)||!Contains(node,primitive.B)||!Contains(node,primitive.C))
                        return fail("Ray leaf bounds exclude one of its triangles");
                }
            }
            else
            {
                if (link[0]!=index+1 || link[0]>=link[3] || link[1])
                    return fail("Ray internal node does not enter its contiguous subtree");
                const auto right=scene.Nodes[link[0]].Links[3];
                if (right<=link[0] || right>=link[3] || scene.Nodes[right].Links[3]!=link[3])
                    return fail("Ray internal node has inconsistent child escapes");
                for (const auto child:{link[0],right})
                    if (!Contains(node,scene.Nodes[child].Minimum)||!Contains(node,scene.Nodes[child].Maximum))
                        return fail("Ray internal bounds exclude a child subtree");
            }
        }
        if (nextTriangle!=scene.Triangles.size()) return fail("Ray leaves do not partition the triangle array");
        return ValidateRaySceneGeometry(scene.Triangles,scene.Materials,scene.CoverageTexels,error);
    }

    bool SceneRayQuery::Export(RaySceneUpload& output,std::string& error) const
    {
        error.clear(); RaySceneUpload candidate;
        const auto& acceleration=Geometry.GetAcceleration();
        const auto& nodes=acceleration.GetNodes(); const auto& order=acceleration.GetPrimitiveIndices();
        if (nodes.size()>UINT32_MAX || order.size()>UINT32_MAX || Meshes.size()>UINT32_MAX)
        { error="Ray scene exceeds 32-bit shader addressing"; return false; }
        candidate.Nodes.resize(nodes.size()); candidate.Triangles.reserve(order.size()); candidate.Materials.reserve(Meshes.size());
        // Reuse immutable source images, but distinguish their channels. A map
        // shared by many instances is uploaded once; no authored asset is added.
        constexpr unsigned PixelChannels=4;
        std::unordered_map<const ImageData*,std::array<std::array<std::uint32_t,4>,PixelChannels>> images;
        auto image=[&](const std::shared_ptr<const ImageData>& source,unsigned channel,std::array<std::uint32_t,4>& descriptor) {
            descriptor={}; if (!source || !source->IsValid()) return true;
            auto& packed=images[source.get()][channel];
            if (packed[3]) { descriptor=packed; return true; }
            const std::uint64_t count=std::uint64_t(source->Width)*source->Height;
            if (count>UINT32_MAX-candidate.CoverageTexels.size()) { error="Ray coverage atlas exceeds 32-bit addressing"; return false; }
            descriptor={static_cast<std::uint32_t>(candidate.CoverageTexels.size()),source->Width,source->Height,RaySceneImagePresent};
            constexpr float ByteMaximum=255;
            for (std::size_t texel=0;texel<count;++texel)
                candidate.CoverageTexels.push_back(source->PixelsRGBA8[texel*PixelChannels+channel]/ByteMaximum);
            packed=descriptor; return true;
        };
        for (const auto& mesh:Meshes)
        {
            RaySceneMaterial material;
            material.Coverage={mesh.Color.W*mesh.Material.BaseColor.W,AlphaTesting ? std::max(mesh.Material.AlphaCutoff,0.0f) : 0,0,0};
            if (material.Coverage.Y>0 &&
                ((TextureMapping && !image(mesh.Material.DiffuseImage,3,material.DiffuseAlpha)) ||
                    !image(mesh.Material.OpacityImage,0,material.OpacityRed))) return false;
            candidate.Materials.push_back(material);
        }
        const auto& indices=Geometry.GetIndices();
        for (const auto source:order)
        {
            const auto first=std::size_t(source)*3; const auto material=TriangleMeshes[source];
            const auto& a=Vertices[indices[first]]; const auto& b=Vertices[indices[first+1]]; const auto& c=Vertices[indices[first+2]];
            RaySceneTriangle triangle; triangle.A=PositionOpacity(a); triangle.B=PositionOpacity(b); triangle.C=PositionOpacity(c);
            triangle.UVAB={a.TexCoord.X,a.TexCoord.Y,b.TexCoord.X,b.TexCoord.Y}; triangle.UVC={c.TexCoord.X,c.TexCoord.Y,0,0};
            triangle.Metadata={material,source,Meshes[material].SourceIndex,Meshes[material].CastShadow ? RaySceneShadowCaster : 0};
            candidate.Triangles.push_back(triangle);
        }
        std::vector<std::pair<std::uint32_t,std::uint32_t>> pending;
        if (!nodes.empty()) pending.emplace_back(0,static_cast<std::uint32_t>(nodes.size()));
        while (!pending.empty())
        {
            const auto [index,escape]=pending.back(); pending.pop_back(); const auto& node=nodes[index]; auto& target=candidate.Nodes[index];
            target.Minimum={node.Bounds.Min.x,node.Bounds.Min.y,node.Bounds.Min.z,0};
            target.Maximum={node.Bounds.Max.x,node.Bounds.Max.y,node.Bounds.Max.z,0};
            if (node.IsLeaf()) target.Links={RaySceneInvalidIndex,node.FirstPrimitive,node.PrimitiveCount,escape};
            else
            {
                target.Links={node.Left,0,0,escape};
                pending.emplace_back(node.Right,escape); pending.emplace_back(node.Left,node.Right);
            }
        }
        if (!ValidateRaySceneUpload(candidate,error)) return false;
        output=std::move(candidate); return true;
    }
}
