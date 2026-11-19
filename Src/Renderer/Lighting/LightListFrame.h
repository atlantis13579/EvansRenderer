#pragma once

#include "Renderer/Lighting/ClusteredForwardContract.h"

namespace Evans::Renderer
{
    // CPU ownership for one native frame. GPU synchronization belongs to each
    // API, but camera and light identity must agree before any light-index read.
    class LightListFrameState
    {
    public:
        void Reset() noexcept { Frame = {}; Assignment = {}; Clusters = {}; }
        bool Prepare(const LightListFrameDesc& frame, std::string& error)
        {
            error.clear();
            if (Prepared() || !frame.Lights || frame.Layout > LightListLayout::Clusters)
            { error = "Light lists require one complete scene snapshot and a known layout per frame"; return false; }
            ShadowProjection projection;
            if (!BuildShadowProjection(frame.ViewProjection, projection, error)) return false;
            Frame = frame;
            return true;
        }
        bool CompleteAssignment(const ForwardPlusGrid& grid, std::string& error)
        {
            error.clear();
            if (!Prepared() || Assigned() || Frame.Layout != LightListLayout::Tiles)
            { error = "Tiled assignment requires a prepared tiled frame and executes once"; return false; }
            LightListSurfaceConstants constants;
            if (!BuildForwardPlusSurfaceConstants(*Frame.Lights, LightListSelection::Opaque, &grid, constants, error)) return false;
            Assignment = grid;
            return true;
        }
        bool CompleteAssignment(const ClusteredForwardGrid& grid, std::string& error)
        {
            error.clear();
            if (!Prepared() || Assigned() || Frame.Layout != LightListLayout::Clusters)
            { error = "Clustered assignment requires a prepared clustered frame and executes once"; return false; }
            LightListSurfaceConstants constants;
            if (!BuildClusteredForwardSurfaceConstants(*Frame.Lights, LightListSelection::Opaque, &grid, constants, error)) return false;
            Clusters = grid;
            return true;
        }
        bool ValidateDraw(const LightingState& lighting, const Matrix4& camera, bool mainCamera, std::string& error) const
        {
            error.clear();
            LightingState expected; expected.SceneLights = Frame.Lights;
            if (!Prepared() || !SameSceneLights(lighting, expected))
            { error = "Light-list draws must retain the frame's complete scene-light index space"; return false; }
            if (mainCamera && camera.Values != Frame.ViewProjection.Values)
            { error = "Light-list main-view geometry must use the assignment camera"; return false; }
            return true;
        }
        bool Surface(const LightingState& lighting, const Matrix4& camera, LightListSelection selection,
            LightListSurfaceConstants& constants, std::string& error) const
        {
            constants = {};
            const bool mainCamera = selection != LightListSelection::FullScene;
            if (!ValidateDraw(lighting, camera, mainCamera, error)) return false;
            if (Frame.Layout == LightListLayout::Clusters)
                return BuildClusteredForwardSurfaceConstants(*Frame.Lights, selection, Assigned() ? &Clusters : nullptr, constants, error);
            return BuildForwardPlusSurfaceConstants(*Frame.Lights, selection, Assigned() ? &Assignment : nullptr, constants, error);
        }
        bool Prepared() const noexcept { return Frame.Lights != nullptr; }
        bool Assigned() const noexcept { return Assignment.TileCount != 0 || Clusters.ClusterCount != 0; }
        const LightListFrameDesc& Description() const noexcept { return Frame; }
        const ForwardPlusGrid& Grid() const noexcept { return Assignment; }
        const ClusteredForwardGrid& ClusterGrid() const noexcept { return Clusters; }
    private:
        LightListFrameDesc Frame;
        ForwardPlusGrid Assignment;
        ClusteredForwardGrid Clusters;
    };
}
