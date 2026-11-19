#pragma once

#include "Renderer/Pipelines/VisibilityBuffer.h"
#include <algorithm>

namespace Evans::Renderer
{
    // Each bounded group builds a deterministic local position index, applies
    // the camera transform once per unique vertex, and compacts primitives.
    // Primitive attributes retain references and homogeneous interpolation planes
    // across UV/color seams, without a fragment-barycentric device requirement.
    // Microsoft Mesh Shader specification; VK_EXT_mesh_shader; Metal MSL 3.0.
    inline constexpr unsigned MeshShaderThreads=32,MeshShaderTriangles=32;
    inline constexpr unsigned MeshShaderVertices=MeshShaderTriangles*VisibilityTriangleCorners;
    inline constexpr unsigned MeshShaderColumns=256,MeshShaderArgumentWords=3;
    inline constexpr unsigned MeshShaderPayloadBytes=MeshShaderThreads*sizeof(unsigned);
    // Auto admits a task stage once a full candidate batch is available. This
    // overhead heuristic is not a universal crossover; GUI modes allow device
    // comparisons. Unsupported task stages always use direct mesh.
    inline constexpr unsigned MeshShaderTaskMinimumTriangles=MeshShaderTriangles*MeshShaderThreads;
    [[nodiscard]] inline bool UseMeshTaskStage(Techniques::MeshShaderDispatchMode mode,unsigned triangles,bool supported)
    {
        return supported && mode!=Techniques::MeshShaderDispatchMode::Direct &&
            (mode==Techniques::MeshShaderDispatchMode::Task || triangles>=MeshShaderTaskMinimumTriangles);
    }
    inline constexpr unsigned MeshShaderHashSlots=256,MeshShaderVertexMaskWords=(MeshShaderVertices+MeshShaderThreads-1)/MeshShaderThreads;
    inline constexpr unsigned MeshShaderSharedBytes=MeshShaderVertices*sizeof(Float4)*3+
        (MeshShaderHashSlots+MeshShaderVertexMaskWords+1)*sizeof(unsigned);
    static_assert(MeshShaderTriangles==MeshShaderThreads && MeshShaderThreads==32 &&
        MeshShaderHashSlots>MeshShaderVertices && (MeshShaderHashSlots&(MeshShaderHashSlots-1))==0,"Indexed mesh workgroup bounds changed");
    struct MeshShaderGrid { unsigned X=0,Y=1,Z=1; };
    static_assert(sizeof(MeshShaderGrid)==MeshShaderArgumentWords*sizeof(unsigned),"Mesh dispatch ABI changed");
    [[nodiscard]] inline MeshShaderGrid BuildMeshShaderGrid(unsigned triangles)
    {
        const auto groups=triangles/MeshShaderTriangles+unsigned(triangles%MeshShaderTriangles!=0);
        return {groups<MeshShaderColumns?groups:MeshShaderColumns,
            groups?groups/MeshShaderColumns+unsigned(groups%MeshShaderColumns!=0):1,1};
    }
    [[nodiscard]] inline MeshShaderGrid BuildMeshTaskGrid(unsigned triangles)
    { return BuildMeshShaderGrid(triangles/MeshShaderTriangles+unsigned(triangles%MeshShaderTriangles!=0)); }

    // Append one typed record per global 32-triangle cluster to the immutable
    // GPU table. Position[0/1] contain its world AABB; all other fields are zero.
    // Sharing the existing record/binding keeps the ordinary raster ABI intact.
    // A draw-relative group can overlap two global clusters; shaders test both.
    [[nodiscard]] inline bool BuildMeshShaderBounds(const VisibilityScene& scene,
        std::vector<VisibilityTriangle>& bounds,std::string& error)
    {
        const auto count=scene.Triangles.size();
        const auto clusters=count/MeshShaderTriangles+std::size_t(count%MeshShaderTriangles!=0);
        if (count>VisibilityStorageCapacity || clusters>VisibilityStorageCapacity-count)
        { error="Mesh bounds exceed visibility table addressing"; return false; }
        bounds.assign(clusters,{});
        for (std::size_t cluster=0;cluster<clusters;++cluster)
        {
            const auto first=cluster*MeshShaderTriangles,last=std::min(count,first+MeshShaderTriangles);
            auto low=scene.Triangles[first].Position[0],high=low;
            for (auto triangle=first;triangle<last;++triangle)
                for (const auto p:scene.Triangles[triangle].Position)
                {
                    low.X=std::min(low.X,p.X); low.Y=std::min(low.Y,p.Y); low.Z=std::min(low.Z,p.Z);
                    high.X=std::max(high.X,p.X); high.Y=std::max(high.Y,p.Y); high.Z=std::max(high.Z,p.Z);
                }
            bounds[cluster].Position[0]=low; bounds[cluster].Position[1]=high;
        }
        return true;
    }
}
