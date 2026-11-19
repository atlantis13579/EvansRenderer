#include "Renderer/Nanite/NaniteRaster.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Evans::Renderer
{
    bool BuildNaniteVisibilityScene(const std::vector<VisibilityNativeMeshView>& meshes,VisibilityScene& output,std::string& error)
    {
        VisibilityScene candidate;
        if (meshes.size()>UINT32_MAX) { error="Native visibility exceeds draw addressing"; return false; }
        for (const auto& mesh:meshes)
        {
            if (!mesh.Visible || !mesh.IndexCount) continue;
            const auto triangles=mesh.IndexCount/VisibilityTriangleCorners;
            if (mesh.IndexCount%VisibilityTriangleCorners || triangles>NaniteRasterMaximumTriangles-candidate.NativeTriangleCount)
            { error="Native visibility source has incomplete triangles or exceeds the raster budget"; return false; }
            const unsigned material=unsigned(candidate.NativeMeshes.size());
            candidate.Draws.push_back({candidate.NativeTriangleCount,unsigned(triangles),material,RealtimeWorldIsMirrored(mesh.World)});
            candidate.NativeMeshes.push_back(mesh); candidate.NativeTriangleCount+=unsigned(triangles);
        }
        if (!ValidateVisibilityScene(candidate,error)) return false;
        output=std::move(candidate); return true;
    }
    bool BuildNaniteVisibilityGeometryConstants(const VisibilityScene& scene,unsigned source,
        const BufferDesc& vertices,const BufferDesc& indices,NaniteVisibilityGeometryConstants& output,unsigned& groupsX,unsigned& groupsY,std::string& error)
    {
        if (source>=scene.NativeMeshes.size() || source>=scene.Draws.size() || scene.TriangleCount()>NaniteRasterMaximumTriangles ||
            vertices.Usage!=BufferUsage::Vertex || indices.Usage!=BufferUsage::Index || !vertices.SizeInBytes ||
            vertices.SizeInBytes%sizeof(Vertex) || indices.SizeInBytes%sizeof(unsigned) ||
            vertices.SizeInBytes>UINT32_MAX || indices.SizeInBytes>UINT32_MAX)
        { error="Native visibility requires compatible decoded vertex/index allocations"; return false; }
        const auto& mesh=scene.NativeMeshes[source]; const auto& range=scene.Draws[source];
        if (!scene.Triangles.empty() || mesh.IndexCount!=std::size_t(range.TriangleCount)*VisibilityTriangleCorners ||
            mesh.FirstIndex>indices.SizeInBytes/sizeof(unsigned) || mesh.IndexCount>indices.SizeInBytes/sizeof(unsigned)-mesh.FirstIndex ||
            !range.TriangleCount || range.FirstTriangle>scene.TriangleCount() || range.TriangleCount>scene.TriangleCount()-range.FirstTriangle)
        { error="Native visibility source exceeds its GPU allocation"; return false; }
        for (float value:mesh.World.Values) if (!std::isfinite(value)) { error="Native visibility transform is not finite"; return false; }
        const auto& m=mesh.World.Values;
        if (m[12]!=0 || m[13]!=0 || m[14]!=0 || m[15]!=1) { error="Native visibility transform must be affine"; return false; }
        NaniteVisibilityGeometryConstants c{}; c.World=mesh.World; c.Normal=BuildRealtimeNormalMatrix(mesh.World);
        for (float value:c.Normal.Values) if (!std::isfinite(value)) { error="Native visibility normal transform is not finite"; return false; }
        const unsigned groups=(range.TriangleCount+NaniteRasterThreads-1)/NaniteRasterThreads;
        c.Range={range.FirstTriangle,range.TriangleCount,mesh.FirstIndex,unsigned(vertices.SizeInBytes/sizeof(Vertex))};
        c.Dispatch={std::min(groups,NaniteRasterDispatchColumns),range.TriangleCount,range.Material,0};
        groupsX=c.Dispatch.X; groupsY=(groups+groupsX-1)/groupsX; output=c; error.clear(); return true;
    }
    bool BuildNaniteFrameRasterWork(const VisibilityBufferDesc& frame,unsigned width,unsigned height,
        NaniteFrameRasterWork& output,std::string& error)
    {
        if (!frame.NaniteRaster || !width || !height || width>NaniteRasterMaximumSide || height>NaniteRasterMaximumSide ||
            !frame.MaximumSoftwareExtent || frame.MaximumSoftwareExtent>32 || frame.SubpixelBits<4 || frame.SubpixelBits>8)
        { error="Nanite frame raster dimensions or precision are unsupported"; return false; }
        if (!ValidateVisibilityFrame(frame,int(width),int(height),error)) return false;
        if (frame.Scene->TriangleCount()>NaniteRasterMaximumTriangles)
        { error="Nanite frame raster exceeds the triangle budget"; return false; }
        const std::size_t pixels=std::size_t(width)*height,triangles=frame.Scene->TriangleCount();
        const auto inputWords=2*pixels+triangles*sizeof(NaniteRasterTriangle)/sizeof(unsigned),outputWords=2*pixels+triangles+(frame.NaniteVisibility?pixels:0);
        if ((inputWords+outputWords)*sizeof(unsigned)>NaniteRasterMaximumBytes)
        { error="Nanite frame raster scratch exceeds the memory budget"; return false; }
        if (bool(frame.NaniteVisibility)!=!frame.NaniteClusters.empty() || frame.NaniteClusters.size()>NaniteVisibilityMaximumQueries)
        { error="Nanite cluster ranges require a matching visibility batch"; return false; }
        NaniteFrameRasterWork result; result.InputWords=unsigned(inputWords); result.OutputWords=unsigned(outputWords);
        NaniteFrameRasterConstants constants{}; auto& c=constants.Raster;
        c.Counts={width,height,unsigned(triangles),unsigned(pixels)};
        c.Layout={0,unsigned(pixels),unsigned(2*pixels),unsigned(2*pixels)};
        c.Quality={frame.MaximumSoftwareExtent,frame.SubpixelBits,unsigned(NaniteVisibilityState::MainVisible),frame.NaniteVisibility?NaniteRasterPriorityOrdering:0u}; constants.ViewProjection=frame.ViewProjection;
        const auto append=[&](unsigned pass,unsigned invocations,unsigned phase)
        {
            if (!invocations) return;
            const unsigned groups=(invocations+NaniteRasterThreads-1)/NaniteRasterThreads;
            c.Dispatch={std::min(groups,NaniteRasterDispatchColumns),invocations,phase,0};
            result.Dispatches.push_back({constants,pass,c.Dispatch.X,(groups+c.Dispatch.X-1)/c.Dispatch.X});
        };
        // Classify before hardware submission using the same raster kernel.
        // Input and route buffers remain on the GPU throughout the frame.
        append(3,unsigned(inputWords),0);
        std::size_t cluster=0; unsigned previousEnd=0;
        for (const auto& draw:frame.Draws)
        {
            const auto& range=frame.Scene->Draws[draw.Range];
            if (frame.NaniteVisibility && range.FirstTriangle<previousEnd)
            { error="Occlusion ranges must retain triangle-reference order for depth ties"; return false; }
            previousEnd=range.FirstTriangle+range.TriangleCount;
            const unsigned flags=(draw.Pipeline.BackFaceCulling?NaniteRasterCullBackFace:0u)|
                (range.Mirrored?NaniteRasterFrontClockwise:0u)|(draw.Pipeline.AlphaCutoff>0?NaniteRasterForceHardware:0u);
            if (!frame.NaniteVisibility)
            { constants.Range={range.FirstTriangle,range.TriangleCount,flags,0}; append(4,range.TriangleCount,0); }
            else
            {
                unsigned covered=0;
                while (cluster<frame.NaniteClusters.size() && frame.NaniteClusters[cluster].Range==draw.Range)
                {
                    const auto& part=frame.NaniteClusters[cluster];
                    if (!part.TriangleCount || part.FirstTriangle!=range.FirstTriangle+covered || part.TriangleCount>range.TriangleCount-covered)
                    { error="Nanite clusters do not partition their material range"; return false; }
                    constants.Range={part.FirstTriangle,part.TriangleCount,flags,unsigned(++cluster)}; append(4,part.TriangleCount,0); covered+=part.TriangleCount;
                }
                if (covered!=range.TriangleCount) { error="Nanite cluster coverage is incomplete"; return false; }
            }
        }
        if (cluster!=frame.NaniteClusters.size()) { error="Nanite clusters reference an unsubmitted material"; return false; }
        constants.Range={}; append(1,unsigned(triangles),4);
        result.HardwareBoundary=unsigned(result.Dispatches.size());
        append(5,unsigned(pixels),0);
        append(0,unsigned(pixels),0); append(1,unsigned(triangles),1);
        append(2,unsigned(pixels),2); append(1,unsigned(triangles),3);
        output=std::move(result); error.clear(); return true;
    }
    bool BuildNaniteRasterDrawConstants(const VisibilityBufferDesc& frame,const VisibilityMaterialDraw* draw,
        unsigned width,unsigned height,NaniteRasterDrawConstants& output)
    {
        if (!frame.Scene || !width || !height || width>NaniteRasterMaximumSide || height>NaniteRasterMaximumSide ||
            frame.Scene->TriangleCount()>NaniteRasterMaximumTriangles || !frame.MaximumSoftwareExtent || frame.MaximumSoftwareExtent>32 ||
            frame.SubpixelBits<4 || frame.SubpixelBits>8 || (draw && draw->Range>=frame.Scene->Draws.size())) return false;
        NaniteRasterDrawConstants c{};
        const unsigned pixels=width*height;
        c.Raster.Counts={width,height,unsigned(frame.Scene->TriangleCount()),pixels};
        c.Raster.Layout={0,pixels,2*pixels,2*pixels};
        c.Raster.Quality={frame.MaximumSoftwareExtent,frame.SubpixelBits,0,frame.NaniteVisibility?NaniteRasterPriorityOrdering:0u};
        if (draw)
        {
            VisibilityConstants material; const auto& range=frame.Scene->Draws[draw->Range];
            if (!BuildVisibilityConstants(frame.ViewProjection,draw->Pipeline,int(width),int(height),frame.Scene->TriangleCount(),range,
                bool(draw->Pipeline.DiffuseTexture),bool(draw->Pipeline.NormalTexture),bool(draw->Pipeline.OpacityTexture),material)) return false;
            c.Range={range.FirstTriangle,range.TriangleCount,0,0}; c.Material={material.Material.Y,material.Material.Z,0,0};
            c.Images={draw->Pipeline.DiffuseTexture?1u:0u,draw->Pipeline.OpacityTexture?1u:0u,0,0};
        }
        output=c; return true;
    }
}
