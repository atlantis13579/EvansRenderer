#pragma once

#include "Renderer/Vulkan/VulkanShaderCompiler.h"
#include "Renderer/TechniqueShader.h"

#include <iterator>

namespace Evans::Renderer
{
inline constexpr VulkanShaderProgram VulkanMeshShaders[7]{
    {"MeshShaders.glsl",VulkanShaderStage::Mesh},
    {"MeshShaders.glsl",VulkanShaderStage::Mesh,"EVANS_MESH_NANITE=1"},
    {"MeshShaders.glsl",VulkanShaderStage::Compute,"EVANS_MESH_ARGUMENTS=1"},
    {"MeshShaders.glsl",VulkanShaderStage::Mesh,"EVANS_MESH_NATIVE_BARYCENTRICS=1"},
    {"MeshShaders.glsl",VulkanShaderStage::Task,"EVANS_MESH_TASK_STAGE=1"},
    {"MeshShaders.glsl",VulkanShaderStage::Mesh,"EVANS_MESH_TASK_INPUT=1"},
    {"MeshShaders.glsl",VulkanShaderStage::Mesh,"EVANS_MESH_TASK_INPUT=1;EVANS_MESH_NATIVE_BARYCENTRICS=1"}};
inline constexpr VulkanShaderProgram VulkanMeshFragments[4]{
    {"VisibilityBuffer.glsl",VulkanShaderStage::MeshFragment,"EVANS_VISIBILITY_MESH=1"},
    {"NaniteHardware.glsl",VulkanShaderStage::MeshFragment,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=0;EVANS_HARDWARE_MESH=1"},
    {"NaniteHardware.glsl",VulkanShaderStage::MeshFragment,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=1;EVANS_HARDWARE_MESH=1"},
    {"VisibilityBuffer.glsl",VulkanShaderStage::MeshFragment,"EVANS_VISIBILITY_MESH=1;EVANS_MESH_NATIVE_BARYCENTRICS=1"}};
inline constexpr VulkanShaderProgram VulkanLumenRadianceTraceShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_RADIANCE_TRACE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenProbeIntegrateShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_PROBE_INTEGRATE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenProbeTraceShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_PROBE_TRACE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenScreenProbeShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_SCREEN_PROBES_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenReflectionResolveShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_REFLECTION_RESOLVE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenReflectionRayShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_REFLECTION_RAYS_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenWorldLookupShader{"LumenWorldCache.glsl",VulkanShaderStage::Compute};
inline constexpr VulkanShaderProgram VulkanLumenHardwareShaders[6]{
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_DISTANCE_TRACE_STAGE=1;EVANS_LUMEN_HARDWARE_RAYS=1"},
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_RADIANCE_TRACE_STAGE=1;EVANS_LUMEN_HARDWARE_RAYS=1"},
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_PROBE_TRACE_STAGE=1;EVANS_LUMEN_HARDWARE_RAYS=1"},
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_LIGHTING_STAGE=1;EVANS_LUMEN_HARDWARE_RAYS=1"},
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_LIGHTING_STAGE=1;EVANS_LUMEN_RADIOSITY=1;EVANS_LUMEN_HARDWARE_RAYS=1"},
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_GATHER_VISIBILITY_STAGE=1;EVANS_LUMEN_HARDWARE_RAYS=1"}};
inline constexpr VulkanShaderProgram VulkanLumenHairIndirectShader{"LumenHair.glsl",VulkanShaderStage::Fragment};
inline constexpr VulkanShaderProgram VulkanLumenHairCompositeShaders[2]{
    {"LumenHair.glsl",VulkanShaderStage::Vertex,"EVANS_LUMEN_HAIR_COMPOSITE=1;EVANS_VERTEX_STAGE=1"},
    {"LumenHair.glsl",VulkanShaderStage::Fragment,"EVANS_LUMEN_HAIR_COMPOSITE=1"}};
inline constexpr VulkanShaderProgram VulkanLumenVolumeSupportShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_VOLUME_VISIBILITY_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenScreenDepthShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_SCREEN_DEPTH_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenScreenTraceShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_SCREEN_TRACE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenRadianceResolveShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_RADIANCE_RESOLVE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenSurfaceLightingShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_LIGHTING_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenSurfaceRadiosityShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_LIGHTING_STAGE=1;EVANS_LUMEN_RADIOSITY=1"};
inline constexpr VulkanShaderProgram VulkanLumenSurfaceLookupShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_LOOKUP_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenGlobalDistanceShader{"LumenGlobalDistance.glsl",VulkanShaderStage::Compute};
inline constexpr VulkanShaderProgram VulkanLumenDistanceTraceShader{"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_DISTANCE_TRACE_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLumenSurfaceShaders[3]{
    {"Lumen.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"Lumen.glsl",VulkanShaderStage::Fragment},
    {"Lumen.glsl",VulkanShaderStage::Compute,"EVANS_LUMEN_READ_STAGE=1"}};
// Each descriptor names one independent Vulkan GLSL stage and its compile-time
// options. The runtime compiler loads the file on demand; no SPIR-V is stored
// here. Material permutations retain the same indices as their native passes.
inline constexpr VulkanShaderProgram VulkanVirtualShadowRequestShaders[2]{
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_SHADOW_PASS=0"},
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_SHADOW_PASS=1"}};
inline constexpr VulkanShaderProgram VulkanVirtualAtlasShaders[5]{
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Vertex,"EVANS_VIRTUAL_SHADOW_RASTER_STAGE=1;EVANS_VERTEX_STAGE=1"},
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Fragment,"EVANS_VIRTUAL_SHADOW_RASTER_STAGE=1"},
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Vertex,"EVANS_VIRTUAL_SHADOW_COPY_STAGE=1;EVANS_VERTEX_STAGE=1"},
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Fragment,"EVANS_VIRTUAL_SHADOW_COPY_STAGE=1"},
    {"VirtualShadowMaps.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_SHADOW_READ_STAGE=1"}};

inline constexpr VulkanShaderProgram VulkanVirtualShadowLookup{"VirtualShadowMaps.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_SHADOW_LOOKUP_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanTSRCompute{"TSR.glsl",VulkanShaderStage::Compute};
inline constexpr VulkanShaderProgram VulkanTSRPack{"TSR.glsl",VulkanShaderStage::Compute,"EVANS_TSR_PACK_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanTSRExport[2]{
    {"TSR.glsl",VulkanShaderStage::Vertex,"EVANS_TSR_EXPORT_VERTEX_STAGE=1"},
    {"TSR.glsl",VulkanShaderStage::Fragment,"EVANS_TSR_EXPORT_FRAGMENT_STAGE=1"}};

inline constexpr VulkanShaderProgram VulkanNaniteDraws{"NaniteDraws.glsl",VulkanShaderStage::Compute};
inline constexpr VulkanShaderProgram VulkanNaniteBinShaders[3]{
    {"NaniteDraws.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_BIN_PASS=0"},
    {"NaniteDraws.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_BIN_PASS=1"},
    {"NaniteDraws.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_BIN_PASS=2"}};
inline constexpr VulkanShaderProgram VulkanNaniteVisibilityGeometry{"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_VISIBILITY_GEOMETRY=1"};
inline constexpr VulkanShaderProgram VulkanNaniteFrameRasterCompute[6]{
    {"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_RASTER_PASS=0"},
    {"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_RASTER_PASS=1"},
    {"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_RASTER_PASS=2"},
    {"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_FRAME_PASS=3"},
    {"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_FRAME_PASS=4"},
    {"NaniteRaster.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_FRAME_PASS=5"}};
inline constexpr VulkanShaderProgram VulkanNaniteFrameRasterGraphics[4]{
    {"NaniteRaster.glsl",VulkanShaderStage::Vertex,"EVANS_NANITE_GRAPHICS=1;EVANS_VERTEX_STAGE=1"},
    {"NaniteRaster.glsl",VulkanShaderStage::Fragment,"EVANS_NANITE_GRAPHICS=1"},
    {"NaniteRaster.glsl",VulkanShaderStage::Vertex,"EVANS_NANITE_GRAPHICS=1;EVANS_NANITE_EXPORT=1;EVANS_VERTEX_STAGE=1"},
    {"NaniteRaster.glsl",VulkanShaderStage::Fragment,"EVANS_NANITE_GRAPHICS=1;EVANS_NANITE_EXPORT=1"}};
inline constexpr VulkanShaderProgram VulkanNaniteVisibilityShaders[4] = {
    {"NaniteVisibility.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_VISIBILITY_PASS=0"},
    {"NaniteVisibility.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_VISIBILITY_PASS=1"},
    {"NaniteVisibility.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_VISIBILITY_PASS=2"},
    {"NaniteVisibility.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_VISIBILITY_PASS=3"}};
inline constexpr VulkanShaderProgram VulkanNaniteGeometryShaders[5] = {
    {"NaniteGeometry.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_GEOMETRY_PASS=0"},
    {"NaniteGeometry.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_GEOMETRY_PASS=1"},
    {"NaniteGeometry.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_GEOMETRY_PASS=2"},
    {"NaniteGeometry.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_GEOMETRY_PASS=3"},
    {"NaniteGeometry.glsl",VulkanShaderStage::Compute,"EVANS_NANITE_GEOMETRY_PASS=4"},
};
inline constexpr VulkanShaderProgram VulkanVirtualGeometryShaders[6] = {
    {"VirtualGeometry.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_GEOMETRY_PASS=0"},
    {"VirtualGeometry.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_GEOMETRY_PASS=1"},
    {"VirtualGeometry.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_GEOMETRY_PASS=2"},
    {"VirtualGeometry.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_GEOMETRY_PASS=3"},
    {"VirtualGeometry.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_GEOMETRY_PASS=4"},
    {"VirtualGeometry.glsl",VulkanShaderStage::Compute,"EVANS_VIRTUAL_GEOMETRY_PASS=5"},
};

inline constexpr VulkanShaderProgram VulkanTessellationShaders[3] = {
    {"Tessellation.glsl",VulkanShaderStage::Compute,"EVANS_TESSELLATION_PASS=0"},
    {"Tessellation.glsl",VulkanShaderStage::Compute,"EVANS_TESSELLATION_PASS=1"},
    {"Tessellation.glsl",VulkanShaderStage::Compute,"EVANS_TESSELLATION_PASS=2"},
};

inline constexpr VulkanShaderProgram VulkanDisplacementShaders[2] = {
    {"DisplacementMapping.glsl",VulkanShaderStage::Compute,"EVANS_DISPLACEMENT_PASS=0"},
    {"DisplacementMapping.glsl",VulkanShaderStage::Compute,"EVANS_DISPLACEMENT_PASS=1"},
};

inline constexpr VulkanShaderProgram VulkanParallaxOcclusionShaders[2] = {
    {"ParallaxOcclusionMapping.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"ParallaxOcclusionMapping.glsl",VulkanShaderStage::Fragment},
};

inline constexpr VulkanShaderProgram VulkanTemporalEffectsShaders[2] = {
    {"TemporalEffects.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"TemporalEffects.glsl",VulkanShaderStage::Fragment},
};

// CascadedShadow source variants.
inline constexpr VulkanShaderProgram VulkanCascadedShadowVertex {"CascadedShadowMap.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanCascadedShadowFragment {"CascadedShadowMap.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanCascadedShadowShaders[2] = {
    VulkanCascadedShadowVertex,
    VulkanCascadedShadowFragment,
};

// ClusteredForward source variants.
inline constexpr VulkanShaderProgram VulkanClusteredForwardAssignment {"ClusteredForward.glsl", VulkanShaderStage::Compute, ""};

// ClusteredForwardGraphics source variants.
inline constexpr VulkanShaderProgram VulkanClusteredForwardSurfaceVS {"ClusteredForward.glsl", VulkanShaderStage::Vertex, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter0Variant0 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=0"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter0Variant1 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=0;EVANS_CLUSTERED_FORWARD_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter0Variant2 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=0;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter0Variant3 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=0;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter0Variant4 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=0;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter1Variant0 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter1Variant1 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=1;EVANS_CLUSTERED_FORWARD_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter1Variant2 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=1;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter1Variant3 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter1Variant4 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=1;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter2Variant0 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=2"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter2Variant1 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=2;EVANS_CLUSTERED_FORWARD_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter2Variant2 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=2;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter2Variant3 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=2;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter2Variant4 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=2;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter3Variant0 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=3"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter3Variant1 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=3;EVANS_CLUSTERED_FORWARD_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter3Variant2 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=3;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter3Variant3 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=3;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanClusteredForwardFilter3Variant4 {"ClusteredForward.glsl", VulkanShaderStage::Fragment, "EVANS_CLUSTERED_FORWARD_SURFACE=1;EVANS_CLUSTERED_FORWARD_FILTER=3;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};

inline constexpr VulkanShaderProgram VulkanClusteredForwardVertexShader = VulkanClusteredForwardSurfaceVS;
inline constexpr VulkanShaderProgram VulkanClusteredForwardFragmentShaders[4][5] = {{VulkanClusteredForwardFilter0Variant0, VulkanClusteredForwardFilter0Variant1, VulkanClusteredForwardFilter0Variant2, VulkanClusteredForwardFilter0Variant3, VulkanClusteredForwardFilter0Variant4},
{VulkanClusteredForwardFilter1Variant0, VulkanClusteredForwardFilter1Variant1, VulkanClusteredForwardFilter1Variant2, VulkanClusteredForwardFilter1Variant3, VulkanClusteredForwardFilter1Variant4},
{VulkanClusteredForwardFilter2Variant0, VulkanClusteredForwardFilter2Variant1, VulkanClusteredForwardFilter2Variant2, VulkanClusteredForwardFilter2Variant3, VulkanClusteredForwardFilter2Variant4},
{VulkanClusteredForwardFilter3Variant0, VulkanClusteredForwardFilter3Variant1, VulkanClusteredForwardFilter3Variant2, VulkanClusteredForwardFilter3Variant3, VulkanClusteredForwardFilter3Variant4}};

// Contact source variants.
inline constexpr VulkanShaderProgram VulkanContactCompute {"ContactShadows.glsl", VulkanShaderStage::Compute, ""};
inline constexpr VulkanShaderProgram VulkanContactComputeMS {"ContactShadows.glsl", VulkanShaderStage::Compute, "EVANS_MULTISAMPLE=1"};

// ContactGraphics source variants.
inline constexpr VulkanShaderProgram VulkanContactDepthVS {"ContactShadows.glsl", VulkanShaderStage::Vertex, "EVANS_CONTACT_DEPTH=1;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanContactDepthFS {"ContactShadows.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanContactBasePhongMS0 {"Phong.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseFlatMS0 {"Flat.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseGouraudMS0 {"Gouraud.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanContactBasePBRMS0 {"AreaLights.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseToonMS0 {"Toon.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseDeferredVS {"Deferred.glsl", VulkanShaderStage::Vertex, "EVANS_CONTACT_RECEIVER=1;EVANS_DEFERRED_LIGHTING=1;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseDeferredMS0 {"Deferred.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_DEFERRED_LIGHTING=1"};
inline constexpr VulkanShaderProgram VulkanContactBasePhongMS1 {"Phong.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseFlatMS1 {"Flat.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseGouraudMS1 {"Gouraud.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanContactBasePBRMS1 {"AreaLights.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseToonMS1 {"Toon.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanContactBaseDeferredMS1 {"Deferred.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_DEFERRED_LIGHTING=1"};
inline constexpr VulkanShaderProgram VulkanContactPCSSPhongMS0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=0"};
inline constexpr VulkanShaderProgram VulkanContactPCSSFlatMS0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=1"};
inline constexpr VulkanShaderProgram VulkanContactPCSSGouraudMS0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=2"};
inline constexpr VulkanShaderProgram VulkanContactPCSSPBRMS0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=3"};
inline constexpr VulkanShaderProgram VulkanContactPCSSToonMS0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=4"};
inline constexpr VulkanShaderProgram VulkanContactPCSSDeferredVS {"PCSS.glsl", VulkanShaderStage::Vertex, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=5;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanContactPCSSDeferredMS0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_PCSS_PROGRAM=5"};
inline constexpr VulkanShaderProgram VulkanContactPCSSPhongMS1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_PCSS_PROGRAM=0"};
inline constexpr VulkanShaderProgram VulkanContactPCSSFlatMS1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_PCSS_PROGRAM=1"};
inline constexpr VulkanShaderProgram VulkanContactPCSSGouraudMS1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_PCSS_PROGRAM=2"};
inline constexpr VulkanShaderProgram VulkanContactPCSSPBRMS1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_PCSS_PROGRAM=3"};
inline constexpr VulkanShaderProgram VulkanContactPCSSToonMS1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_PCSS_PROGRAM=4"};
inline constexpr VulkanShaderProgram VulkanContactPCSSDeferredMS1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_PCSS_PROGRAM=6"};
inline constexpr VulkanShaderProgram VulkanContactVSMPhongMS0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=0"};
inline constexpr VulkanShaderProgram VulkanContactVSMFlatMS0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=1"};
inline constexpr VulkanShaderProgram VulkanContactVSMGouraudMS0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=2"};
inline constexpr VulkanShaderProgram VulkanContactVSMPBRMS0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=3"};
inline constexpr VulkanShaderProgram VulkanContactVSMToonMS0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=4"};
inline constexpr VulkanShaderProgram VulkanContactVSMDeferredVS {"VSM.glsl", VulkanShaderStage::Vertex, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=5;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanContactVSMDeferredMS0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_VSM_PROGRAM=5"};
inline constexpr VulkanShaderProgram VulkanContactVSMPhongMS1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_VSM_PROGRAM=0"};
inline constexpr VulkanShaderProgram VulkanContactVSMFlatMS1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_VSM_PROGRAM=1"};
inline constexpr VulkanShaderProgram VulkanContactVSMGouraudMS1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_VSM_PROGRAM=2"};
inline constexpr VulkanShaderProgram VulkanContactVSMPBRMS1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_VSM_PROGRAM=3"};
inline constexpr VulkanShaderProgram VulkanContactVSMToonMS1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_VSM_PROGRAM=4"};
inline constexpr VulkanShaderProgram VulkanContactVSMDeferredMS1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1;EVANS_VSM_PROGRAM=6"};

inline constexpr VulkanShaderProgram VulkanContactDepthShaders[2] = {VulkanContactDepthVS, VulkanContactDepthFS};
inline constexpr VulkanShaderProgram VulkanContactSurfaceShaders[3][2][5] = {{{VulkanContactBasePhongMS0, VulkanContactBaseFlatMS0, VulkanContactBaseGouraudMS0, VulkanContactBasePBRMS0, VulkanContactBaseToonMS0}, {VulkanContactBasePhongMS1, VulkanContactBaseFlatMS1, VulkanContactBaseGouraudMS1, VulkanContactBasePBRMS1, VulkanContactBaseToonMS1}},
{{VulkanContactPCSSPhongMS0, VulkanContactPCSSFlatMS0, VulkanContactPCSSGouraudMS0, VulkanContactPCSSPBRMS0, VulkanContactPCSSToonMS0}, {VulkanContactPCSSPhongMS1, VulkanContactPCSSFlatMS1, VulkanContactPCSSGouraudMS1, VulkanContactPCSSPBRMS1, VulkanContactPCSSToonMS1}},
{{VulkanContactVSMPhongMS0, VulkanContactVSMFlatMS0, VulkanContactVSMGouraudMS0, VulkanContactVSMPBRMS0, VulkanContactVSMToonMS0}, {VulkanContactVSMPhongMS1, VulkanContactVSMFlatMS1, VulkanContactVSMGouraudMS1, VulkanContactVSMPBRMS1, VulkanContactVSMToonMS1}}};
inline constexpr VulkanShaderProgram VulkanContactDeferredShaders[3][2][2] = {{{VulkanContactBaseDeferredVS, VulkanContactBaseDeferredMS0}, {VulkanContactBaseDeferredVS, VulkanContactBaseDeferredMS1}},
{{VulkanContactPCSSDeferredVS, VulkanContactPCSSDeferredMS0}, {VulkanContactPCSSDeferredVS, VulkanContactPCSSDeferredMS1}},
{{VulkanContactVSMDeferredVS, VulkanContactVSMDeferredMS0}, {VulkanContactVSMDeferredVS, VulkanContactVSMDeferredMS1}}};

// Cubemap source variants.
inline constexpr VulkanShaderProgram VulkanCubemapVertex {"CubemapReflections.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanCubemapFragment {"CubemapReflections.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanCubemapMipFragment {"CubemapReflections.glsl", VulkanShaderStage::Fragment, "EVANS_CUBE_MIP=1"};

// Native SPIR-V compiled directly from Src/Shaders/Vulkan/CubemapReflections.glsl.

inline constexpr VulkanShaderProgram VulkanCubemapShaders[2] = {
    VulkanCubemapVertex,
    VulkanCubemapFragment,
};
inline constexpr VulkanShaderProgram VulkanCubemapMipShaders[2] = {
    VulkanCubemapVertex,
    VulkanCubemapMipFragment,
};

// Deferred source variants.
inline constexpr VulkanShaderProgram VulkanDeferredGeometryVertex {"Deferred.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanDeferredGeometryFragment {"Deferred.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanDeferredLightingVertex {"Deferred.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_DEFERRED_LIGHTING=1"};
inline constexpr VulkanShaderProgram VulkanDeferredLightingFragment {"Deferred.glsl", VulkanShaderStage::Fragment, "EVANS_DEFERRED_LIGHTING=1"};
inline constexpr VulkanShaderProgram VulkanDeferredMultisampleVertex {"Deferred.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_DEFERRED_LIGHTING=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanDeferredMultisampleFragment {"Deferred.glsl", VulkanShaderStage::Fragment, "EVANS_DEFERRED_LIGHTING=1;EVANS_MULTISAMPLE=1"};

inline constexpr VulkanShaderProgram VulkanDeferredGeometryShaders[2] = {
    VulkanDeferredGeometryVertex,
    VulkanDeferredGeometryFragment,
};
inline constexpr VulkanShaderProgram VulkanDeferredLightingShaders[2] = {
    VulkanDeferredLightingVertex,
    VulkanDeferredLightingFragment,
};
inline constexpr VulkanShaderProgram VulkanDeferredMultisampleShaders[2] = {
    VulkanDeferredMultisampleVertex,
    VulkanDeferredMultisampleFragment,
};

// DepthOfField source variants.
inline constexpr VulkanShaderProgram VulkanDepthOfFieldVertex {"DepthOfField.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanDepthOfFieldFragment {"DepthOfField.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanDepthOfFieldShaders[2] = {
    VulkanDepthOfFieldVertex,
    VulkanDepthOfFieldFragment,
};

// EnvironmentMap source variants.
inline constexpr VulkanShaderProgram VulkanEnvironmentMapVertex {"EnvironmentMapReflections.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanEnvironmentMapFragment {"EnvironmentMapReflections.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanEnvironmentMapConversionFragment {"EnvironmentMapReflections.glsl", VulkanShaderStage::Fragment, "EVANS_ENVIRONMENT_CONVERSION=1"};
inline constexpr VulkanShaderProgram VulkanEnvironmentMapMipFragment {"EnvironmentMapReflections.glsl", VulkanShaderStage::Fragment, "EVANS_ENVIRONMENT_MIP=1"};

// Native SPIR-V compiled directly from Src/Shaders/Vulkan/EnvironmentMapReflections.glsl.

inline constexpr VulkanShaderProgram VulkanEnvironmentMapShaders[2] = {
    VulkanEnvironmentMapVertex,
    VulkanEnvironmentMapFragment,
};
inline constexpr VulkanShaderProgram VulkanEnvironmentMapConversionShaders[2] = {
    VulkanEnvironmentMapVertex,
    VulkanEnvironmentMapConversionFragment,
};
inline constexpr VulkanShaderProgram VulkanEnvironmentMapMipShaders[2] = {
    VulkanEnvironmentMapVertex,
    VulkanEnvironmentMapMipFragment,
};

// ForwardPlus source variants.
inline constexpr VulkanShaderProgram VulkanTiledDeferredSingle {"TiledDeferred.glsl", VulkanShaderStage::Compute, ""};
inline constexpr VulkanShaderProgram VulkanTiledDeferredMS {"TiledDeferred.glsl", VulkanShaderStage::Compute, "EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusSingle {"ForwardPlus.glsl", VulkanShaderStage::Compute, ""};
inline constexpr VulkanShaderProgram VulkanForwardPlusMS {"ForwardPlus.glsl", VulkanShaderStage::Compute, "EVANS_MULTISAMPLE=1"};

// ForwardPlusGraphics source variants.
inline constexpr VulkanShaderProgram VulkanForwardPlusSurfaceVS {"ForwardPlus.glsl", VulkanShaderStage::Vertex, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter0Variant0 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=0"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter0Variant1 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=0;EVANS_FORWARD_PLUS_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter0Variant2 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=0;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter0Variant3 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=0;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter0Variant4 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=0;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter1Variant0 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter1Variant1 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=1;EVANS_FORWARD_PLUS_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter1Variant2 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=1;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter1Variant3 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter1Variant4 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=1;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter2Variant0 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=2"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter2Variant1 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=2;EVANS_FORWARD_PLUS_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter2Variant2 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=2;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter2Variant3 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=2;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter2Variant4 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=2;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter3Variant0 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=3"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter3Variant1 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=3;EVANS_FORWARD_PLUS_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter3Variant2 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=3;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter3Variant3 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=3;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanForwardPlusFilter3Variant4 {"ForwardPlus.glsl", VulkanShaderStage::Fragment, "EVANS_FORWARD_PLUS_SURFACE=1;EVANS_FORWARD_PLUS_FILTER=3;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};

inline constexpr VulkanShaderProgram VulkanForwardPlusVertexShader = VulkanForwardPlusSurfaceVS;
inline constexpr VulkanShaderProgram VulkanForwardPlusFragmentShaders[4][5] = {{VulkanForwardPlusFilter0Variant0, VulkanForwardPlusFilter0Variant1, VulkanForwardPlusFilter0Variant2, VulkanForwardPlusFilter0Variant3, VulkanForwardPlusFilter0Variant4},
{VulkanForwardPlusFilter1Variant0, VulkanForwardPlusFilter1Variant1, VulkanForwardPlusFilter1Variant2, VulkanForwardPlusFilter1Variant3, VulkanForwardPlusFilter1Variant4},
{VulkanForwardPlusFilter2Variant0, VulkanForwardPlusFilter2Variant1, VulkanForwardPlusFilter2Variant2, VulkanForwardPlusFilter2Variant3, VulkanForwardPlusFilter2Variant4},
{VulkanForwardPlusFilter3Variant0, VulkanForwardPlusFilter3Variant1, VulkanForwardPlusFilter3Variant2, VulkanForwardPlusFilter3Variant3, VulkanForwardPlusFilter3Variant4}};

// TiledDeferredGraphics source variants.
inline constexpr VulkanShaderProgram VulkanTiledDeferredSurfaceVS {"TiledDeferred.glsl", VulkanShaderStage::Vertex, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter0Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=0"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter0Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_TILED_DEFERRED_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter0Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter0Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter0Variant4 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter1Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter1Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_TILED_DEFERRED_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter1Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter1Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter1Variant4 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter2Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=2"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter2Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_TILED_DEFERRED_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter2Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter2Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter2Variant4 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter3Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=3"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter3Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_TILED_DEFERRED_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter3Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter3Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledDeferredFilter3Variant4 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_CONTACT_RECEIVER=1;EVANS_MULTISAMPLE=1"};

inline constexpr VulkanShaderProgram VulkanTiledDeferredVertexShader = VulkanTiledDeferredSurfaceVS;
inline constexpr VulkanShaderProgram VulkanTiledDeferredFragmentShaders[4][5] = {{VulkanTiledDeferredFilter0Variant0, VulkanTiledDeferredFilter0Variant1, VulkanTiledDeferredFilter0Variant2, VulkanTiledDeferredFilter0Variant3, VulkanTiledDeferredFilter0Variant4},
{VulkanTiledDeferredFilter1Variant0, VulkanTiledDeferredFilter1Variant1, VulkanTiledDeferredFilter1Variant2, VulkanTiledDeferredFilter1Variant3, VulkanTiledDeferredFilter1Variant4},
{VulkanTiledDeferredFilter2Variant0, VulkanTiledDeferredFilter2Variant1, VulkanTiledDeferredFilter2Variant2, VulkanTiledDeferredFilter2Variant3, VulkanTiledDeferredFilter2Variant4},
{VulkanTiledDeferredFilter3Variant0, VulkanTiledDeferredFilter3Variant1, VulkanTiledDeferredFilter3Variant2, VulkanTiledDeferredFilter3Variant3, VulkanTiledDeferredFilter3Variant4}};

// Tiled Deferred evaluates the original G-buffer sample with scene-order tile indices.
inline constexpr VulkanShaderProgram VulkanTiledLightingVertex {"TiledDeferred.glsl", VulkanShaderStage::Vertex, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter0Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=0"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter0Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter0Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter0Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=0;EVANS_MULTISAMPLE=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter1Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter1Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter1Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter1Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=1;EVANS_MULTISAMPLE=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter2Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=2"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter2Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter2Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter2Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=2;EVANS_MULTISAMPLE=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter3Variant0 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=3"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter3Variant1 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_MULTISAMPLE=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter3Variant2 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingFilter3Variant3 {"TiledDeferred.glsl", VulkanShaderStage::Fragment, "EVANS_TILED_DEFERRED_SURFACE=1;EVANS_TILED_DEFERRED_LIGHTING=1;EVANS_TILED_DEFERRED_FILTER=3;EVANS_MULTISAMPLE=1;EVANS_CONTACT_RECEIVER=1"};
inline constexpr VulkanShaderProgram VulkanTiledLightingShaders[4][4][2] = {
{{VulkanTiledLightingVertex, VulkanTiledLightingFilter0Variant0}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter0Variant1}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter0Variant2}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter0Variant3}},
{{VulkanTiledLightingVertex, VulkanTiledLightingFilter1Variant0}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter1Variant1}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter1Variant2}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter1Variant3}},
{{VulkanTiledLightingVertex, VulkanTiledLightingFilter2Variant0}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter2Variant1}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter2Variant2}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter2Variant3}},
{{VulkanTiledLightingVertex, VulkanTiledLightingFilter3Variant0}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter3Variant1}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter3Variant2}, {VulkanTiledLightingVertex, VulkanTiledLightingFilter3Variant3}}};

// Gtao source variants.
inline constexpr VulkanShaderProgram VulkanGtaoVertex {"GTAO.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanGtaoFragment {"GTAO.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanGtaoShaders[2] = {
    VulkanGtaoVertex,
    VulkanGtaoFragment,
};

// Hbao source variants.
inline constexpr VulkanShaderProgram VulkanHbaoVertex {"HBAO.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanHbaoFragment {"HBAO.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanHbaoShaders[2] = {
    VulkanHbaoVertex,
    VulkanHbaoFragment,
};

// Hdr source variants.
inline constexpr VulkanShaderProgram VulkanHdrBloomVertex {"Bloom.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanHdrBloomFragment {"Bloom.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanHdrColorGradingVertex {"ColorGrading.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanHdrColorGradingFragment {"ColorGrading.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanHdrToneMappingVertex {"ToneMapping.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanHdrToneMappingFragment {"ToneMapping.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanHdrLensFlareVertex {"LensFlare.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanHdrLensFlareFragment {"LensFlare.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanHdrShaders[][2] = {
    {VulkanHdrBloomVertex, VulkanHdrBloomFragment},
    {VulkanHdrColorGradingVertex, VulkanHdrColorGradingFragment},
    {VulkanHdrToneMappingVertex, VulkanHdrToneMappingFragment},
    {VulkanHdrLensFlareVertex, VulkanHdrLensFlareFragment},
};

// Instancing source variants.
inline constexpr VulkanShaderProgram VulkanInstanceForward {"InstancedRendering.glsl", VulkanShaderStage::Vertex, ""};
inline constexpr VulkanShaderProgram VulkanInstanceGouraud {"InstancedRendering.glsl", VulkanShaderStage::Vertex, "EVANS_INSTANCE_GOURAUD=1"};
inline constexpr VulkanShaderProgram VulkanInstanceDeferred {"InstancedRendering.glsl", VulkanShaderStage::Vertex, "EVANS_INSTANCE_DEFERRED=1"};
inline constexpr VulkanShaderProgram VulkanInstanceShadow {"InstancedRendering.glsl", VulkanShaderStage::Vertex, "EVANS_INSTANCE_SHADOW=1"};
inline constexpr VulkanShaderProgram VulkanInstanceVelocity {"InstancedRendering.glsl", VulkanShaderStage::Vertex, "EVANS_INSTANCE_VELOCITY=1"};
inline constexpr VulkanShaderProgram VulkanInstanceCascade {"InstancedRendering.glsl", VulkanShaderStage::Vertex, "EVANS_INSTANCE_CASCADE=1"};

inline constexpr VulkanShaderProgram VulkanInstanceShaders[] = {
    VulkanInstanceForward,
    VulkanInstanceGouraud,
    VulkanInstanceDeferred,
    VulkanInstanceShadow,
    VulkanInstanceVelocity,
    VulkanInstanceCascade,
};

// LightShaft source variants.
inline constexpr VulkanShaderProgram VulkanLightShaftVertex {"LightShafts.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanLightShaftFragment {"LightShafts.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanLightShaftShaders[2] = {
    VulkanLightShaftVertex,
    VulkanLightShaftFragment,
};

// MotionBlur source variants.
inline constexpr VulkanShaderProgram VulkanMotionBlurGeometryVertex {"MotionVectors.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanMotionBlurGeometryFragment {"MotionVectors.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanMotionBlurFilterVertex {"MotionBlur.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanMotionBlurFilterFragment {"MotionBlur.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanMotionBlurGeometryShaders[2] = {
    VulkanMotionBlurGeometryVertex,
    VulkanMotionBlurGeometryFragment,
};
inline constexpr VulkanShaderProgram VulkanMotionBlurFilterShaders[2] = {
    VulkanMotionBlurFilterVertex,
    VulkanMotionBlurFilterFragment,
};

// MSAA source variants.
inline constexpr VulkanShaderProgram VulkanMSAAColorVertex {"MSAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanMSAAColorFragment {"MSAA.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanMSAADepthVertex {"MSAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_RESOLVE_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanMSAADepthFragment {"MSAA.glsl", VulkanShaderStage::Fragment, "EVANS_RESOLVE_DEPTH=1"};
inline constexpr VulkanShaderProgram VulkanMSAAMaterialVertex {"MSAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_RESOLVE_MATERIAL=1"};
inline constexpr VulkanShaderProgram VulkanMSAAMaterialFragment {"MSAA.glsl", VulkanShaderStage::Fragment, "EVANS_RESOLVE_MATERIAL=1"};
inline constexpr VulkanShaderProgram VulkanMSAAVelocityVertex {"MSAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_RESOLVE_VELOCITY=1"};
inline constexpr VulkanShaderProgram VulkanMSAAVelocityFragment {"MSAA.glsl", VulkanShaderStage::Fragment, "EVANS_RESOLVE_VELOCITY=1"};
inline constexpr VulkanShaderProgram VulkanMSAARestoreVertex {"MSAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_RESTORE_COLOR=1"};
inline constexpr VulkanShaderProgram VulkanMSAARestoreFragment {"MSAA.glsl", VulkanShaderStage::Fragment, "EVANS_RESTORE_COLOR=1"};

inline constexpr VulkanShaderProgram VulkanMSAAShaders[5][2] = {
    {
        VulkanMSAAColorVertex,
        VulkanMSAAColorFragment,
    },
    {
        VulkanMSAADepthVertex,
        VulkanMSAADepthFragment,
    },
    {
        VulkanMSAAMaterialVertex,
        VulkanMSAAMaterialFragment,
    },
    {
        VulkanMSAAVelocityVertex,
        VulkanMSAAVelocityFragment,
    },
    {
        VulkanMSAARestoreVertex,
        VulkanMSAARestoreFragment,
    },
};

// PCSS source variants.
inline constexpr VulkanShaderProgram VulkanPCSSVertex {"PCSS.glsl", VulkanShaderStage::Vertex, "EVANS_PCSS_PROGRAM=5;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment0 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=0"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment1 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=1"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment2 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=2"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment3 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=3"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment4 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=4"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment5 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=5"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment6 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=6"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment7 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=7"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment8 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=0;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment9 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=1;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment10 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=2;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment11 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=3;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanPCSSFragment12 {"PCSS.glsl", VulkanShaderStage::Fragment, "EVANS_PCSS_PROGRAM=4;EVANS_WEIGHTED_TRANSPARENCY=1"};

inline constexpr VulkanShaderProgram VulkanPCSSFragments[13] = {
    VulkanPCSSFragment0,
    VulkanPCSSFragment1,
    VulkanPCSSFragment2,
    VulkanPCSSFragment3,
    VulkanPCSSFragment4,
    VulkanPCSSFragment5,
    VulkanPCSSFragment6,
    VulkanPCSSFragment7,
    VulkanPCSSFragment8,
    VulkanPCSSFragment9,
    VulkanPCSSFragment10,
    VulkanPCSSFragment11,
    VulkanPCSSFragment12,
};
inline constexpr VulkanShaderProgram VulkanPCSSScreenShaders[3][2] = {
    {VulkanPCSSVertex, VulkanPCSSFragments[5]},
    {VulkanPCSSVertex, VulkanPCSSFragments[6]},
    {VulkanPCSSVertex, VulkanPCSSFragments[7]},
};

// Planar source variants.
inline constexpr VulkanShaderProgram VulkanPlanarVertex {"PlanarReflections.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPlanarFragment {"PlanarReflections.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPlanarMipFragment {"PlanarReflections.glsl", VulkanShaderStage::Fragment, "EVANS_PLANAR_MIP=1"};

// Native SPIR-V compiled directly from Src/Shaders/Vulkan/PlanarReflections.glsl.

inline constexpr VulkanShaderProgram VulkanPlanarShaders[2] = {
    VulkanPlanarVertex,
    VulkanPlanarFragment,
};
inline constexpr VulkanShaderProgram VulkanPlanarMipShaders[2] = {
    VulkanPlanarVertex,
    VulkanPlanarMipFragment,
};

// PostProcess source variants.
inline constexpr VulkanShaderProgram VulkanPostFXAAVertex {"FXAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostFXAAFragment {"FXAA.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostVignetteVertex {"Vignette.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostVignetteFragment {"Vignette.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostFilmGrainVertex {"FilmGrain.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostFilmGrainFragment {"FilmGrain.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostScreenSpaceDistortionVertex {"ScreenSpaceDistortion.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostScreenSpaceDistortionFragment {"ScreenSpaceDistortion.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanPostNPRVertex {"NPR.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostNPRFragment {"NPR.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostPixelationVertex {"Pixelation.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostPixelationFragment {"Pixelation.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostColorEdgeOutlineVertex {"ColorEdgeOutline.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostColorEdgeOutlineFragment {"ColorEdgeOutline.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostDynamicDistortionVertex {"DynamicDistortion.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostDynamicDistortionFragment {"DynamicDistortion.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostOilPaintVertex {"OilPaint.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostOilPaintFragment {"OilPaint.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostWatercolorVertex {"Watercolor.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostWatercolorFragment {"Watercolor.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostDepthNormalOutlineVertex {"DepthNormalOutline.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPostDepthNormalOutlineFragment {"DepthNormalOutline.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPostProcessShaders[][2] = {
    {VulkanPostFXAAVertex, VulkanPostFXAAFragment},
    {VulkanPostVignetteVertex, VulkanPostVignetteFragment},
    {VulkanPostFilmGrainVertex, VulkanPostFilmGrainFragment},
    {VulkanPostScreenSpaceDistortionVertex, VulkanPostScreenSpaceDistortionFragment},
    {VulkanPostNPRVertex, VulkanPostNPRFragment},
    {VulkanPostPixelationVertex, VulkanPostPixelationFragment},
    {VulkanPostColorEdgeOutlineVertex, VulkanPostColorEdgeOutlineFragment},
    {VulkanPostDynamicDistortionVertex, VulkanPostDynamicDistortionFragment},
    {VulkanPostOilPaintVertex, VulkanPostOilPaintFragment},
    {VulkanPostWatercolorVertex, VulkanPostWatercolorFragment},
    {VulkanPostDepthNormalOutlineVertex, VulkanPostDepthNormalOutlineFragment},
};
static_assert(std::size(VulkanPostProcessShaders) == PostProcessTechniqueCount);

// Sao source variants.
inline constexpr VulkanShaderProgram VulkanSaoVertex {"SAO.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanSaoFragment {"SAO.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanSaoShaders[2] = {
    VulkanSaoVertex,
    VulkanSaoFragment,
};

// ShadowMapping source variants.
inline constexpr VulkanShaderProgram VulkanShadowMappingVertex {"ShadowMapping.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanShadowMappingFragment {"ShadowMapping.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanShadowMappingShaders[2] = {
    VulkanShadowMappingVertex,
    VulkanShadowMappingFragment,
};

// Ssao source variants.
inline constexpr VulkanShaderProgram VulkanSsaoVertex {"SSAO.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanSsaoFragment {"SSAO.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanSsaoShaders[2] = {
    VulkanSsaoVertex,
    VulkanSsaoFragment,
};

// Ssr source variants.
inline constexpr VulkanShaderProgram VulkanSsrVertex {"SSR.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanSsrFragment {"SSR.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanSsrShaders[2] = {
    VulkanSsrVertex,
    VulkanSsrFragment,
};

// SSGI uses its own readable Vulkan GLSL, compiled in memory at runtime.
inline constexpr VulkanShaderProgram VulkanGIVertex {"SSGI.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanGIFragment {"SSGI.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanGITemporalFragment {"SSGI.glsl", VulkanShaderStage::Fragment, "EVANS_SSGI_TEMPORAL=1"};
inline constexpr VulkanShaderProgram VulkanGIShaders[2] {VulkanGIVertex, VulkanGIFragment};
inline constexpr VulkanShaderProgram VulkanGITemporalShaders[2] {VulkanGIVertex, VulkanGITemporalFragment};

inline constexpr VulkanShaderProgram VulkanBakedGIShaders[2] {
    {"BakedGI.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"BakedGI.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanLightmapShaders[2] {
    {"Lightmaps.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"Lightmaps.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanRayTracedAOShaders[2] {
    {"RayTracedAO.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"RayTracedAO.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanGpuDrivenCull {"GPUDrivenRendering.glsl", VulkanShaderStage::Compute, ""};
inline constexpr VulkanShaderProgram VulkanNaniteHardwareCompute[4] {
    {"NaniteHardware.glsl",VulkanShaderStage::Compute,"EVANS_HARDWARE_COMPUTE=0"},
    {"NaniteHardware.glsl",VulkanShaderStage::Compute,"EVANS_HARDWARE_COMPUTE=1"},
    {"NaniteHardware.glsl",VulkanShaderStage::Compute,"EVANS_HARDWARE_COMPUTE=2"},
    {"NaniteHardware.glsl",VulkanShaderStage::Compute,"EVANS_HARDWARE_COMPUTE=3"}};
inline constexpr VulkanShaderProgram VulkanNaniteHardwareGraphics[5] {
    {"NaniteHardware.glsl",VulkanShaderStage::Vertex,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=0;EVANS_VERTEX_STAGE=1"},
    {"NaniteHardware.glsl",VulkanShaderStage::Vertex,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=2;EVANS_VERTEX_STAGE=1"},
    {"NaniteHardware.glsl",VulkanShaderStage::Fragment,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=0"},
    {"NaniteHardware.glsl",VulkanShaderStage::Fragment,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=1"},
    {"NaniteHardware.glsl",VulkanShaderStage::Fragment,"EVANS_HARDWARE_GRAPHICS=1;EVANS_HARDWARE_PASS=2"}};
inline constexpr VulkanShaderProgram VulkanNaniteInstanceShaders[3] {
    {"NaniteInstances.glsl",VulkanShaderStage::Compute,"EVANS_INSTANCE_PASS=0"},
    {"NaniteInstances.glsl",VulkanShaderStage::Compute,"EVANS_INSTANCE_PASS=1"},
    {"NaniteInstances.glsl",VulkanShaderStage::Compute,"EVANS_INSTANCE_PASS=2"}};
inline constexpr VulkanShaderProgram VulkanNaniteMaterialShaders[4] {
    {"NaniteMaterials.glsl",VulkanShaderStage::Compute,"EVANS_MATERIAL_PASS=0"},
    {"NaniteMaterials.glsl",VulkanShaderStage::Compute,"EVANS_MATERIAL_PASS=1"},
    {"NaniteMaterials.glsl",VulkanShaderStage::Compute,"EVANS_MATERIAL_PASS=2"},
    {"NaniteMaterials.glsl",VulkanShaderStage::Compute,"EVANS_MATERIAL_PASS=3"}};
inline constexpr VulkanShaderProgram VulkanVisibilityBufferShaders[4] {
    {"VisibilityBuffer.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"},
    {"VisibilityBuffer.glsl", VulkanShaderStage::Fragment, ""},
    {"VisibilityBuffer.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_VISIBILITY_RESOLVE=1"},
    {"VisibilityBuffer.glsl", VulkanShaderStage::Fragment, "EVANS_VISIBILITY_RESOLVE=1"},
};
// Camera transport and composition compile independently from the same native GLSL.
inline constexpr VulkanShaderProgram VulkanPathTracingShaders[2] {
    {"PathTracing.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"PathTracing.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanPathTracingComposeShaders[2] {
    {"PathTracing.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"PathTracing.glsl",VulkanShaderStage::Fragment,"EVANS_PATH_COMPOSE=1"}};
inline constexpr VulkanShaderProgram VulkanRayTracedGIShaders[2] {
    {"RayTracedGI.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"RayTracedGI.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanSubsurfaceScatteringShaders[2] {
    {"SubsurfaceScattering.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"SubsurfaceScattering.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanVoxelGIShaders[2] {
    {"VoxelGlobalIllumination.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"VoxelGlobalIllumination.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanHairFurShaders[2] {
    {"HairFur.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"HairFur.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanHairFurDepthShaders[2] {
    VulkanHairFurShaders[0], {"HairFur.glsl",VulkanShaderStage::Fragment,"EVANS_HAIR_DEPTH_ONLY=1"}};
inline constexpr VulkanShaderProgram VulkanHairFurCaptureShaders[2] {
    VulkanHairFurShaders[0], {"HairFur.glsl",VulkanShaderStage::Fragment,"EVANS_HAIR_CAPTURE=1"}};
inline constexpr VulkanShaderProgram VulkanHairFurMomentShaders[2] {
    VulkanHairFurShaders[0], {"HairFur.glsl",VulkanShaderStage::Fragment,"EVANS_HAIR_VSM_SHADOW=1"}};
inline constexpr VulkanShaderProgram VulkanHairFurVelocityShaders[2] {
    VulkanHairFurShaders[0], {"HairFur.glsl",VulkanShaderStage::Fragment,"EVANS_HAIR_VELOCITY=1"}};
inline constexpr VulkanShaderProgram VulkanHairFurContactShaders[2] {
    {"HairFur.glsl",VulkanShaderStage::Fragment,"EVANS_HAIR_CONTACT_RECEIVER=1"},
    {"HairFur.glsl",VulkanShaderStage::Fragment,"EVANS_HAIR_CONTACT_RECEIVER=1;EVANS_HAIR_CONTACT_SAMPLES=1"}};
inline constexpr VulkanShaderProgram VulkanHybridRenderingShaders[2] {
    {"HybridRendering.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"HybridRendering.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanRayTracedReflectionShaders[2] {
    {"RayTracedReflections.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"RayTracedReflections.glsl",VulkanShaderStage::Fragment,""}};

inline constexpr VulkanShaderProgram VulkanOITShaders[6] {
    {"OIT.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1;EVANS_OIT_CAPTURE=1"},
    {"OIT.glsl",VulkanShaderStage::Fragment,"EVANS_OIT_CAPTURE=1"},
    {"OIT.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1;EVANS_OIT_ACCUMULATE=1"},
    {"OIT.glsl",VulkanShaderStage::Fragment,"EVANS_OIT_ACCUMULATE=1"},
    {"OIT.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"OIT.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanRefractionShaders[4] {
    {"Refraction.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1;EVANS_REFRACTION_CAPTURE=1"},
    {"Refraction.glsl",VulkanShaderStage::Fragment,"EVANS_REFRACTION_CAPTURE=1"},
    {"Refraction.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"Refraction.glsl",VulkanShaderStage::Fragment,""}};
inline constexpr VulkanShaderProgram VulkanLightProbeShaders[2] {
    {"LightProbes.glsl",VulkanShaderStage::Vertex,"EVANS_VERTEX_STAGE=1"},
    {"LightProbes.glsl",VulkanShaderStage::Fragment,""}};

// TemporalAA source variants.
inline constexpr VulkanShaderProgram VulkanTemporalAAVertex {"TAA.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanTemporalAAFragment {"TAA.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanTemporalAAShaders[2] = {
    VulkanTemporalAAVertex,
    VulkanTemporalAAFragment,
};

// Technique source variants.
inline constexpr VulkanShaderProgram VulkanPhongVertex {"Phong.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPhongFragment {"Phong.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanFlatVertex {"Flat.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanFlatFragment {"Flat.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanGouraudVertex {"Gouraud.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanGouraudFragment {"Gouraud.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanPBRVertex {"PBR.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanPBRFragment {"PBR.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanToonVertex {"Toon.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanToonFragment {"Toon.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanAreaLightsVertex {"AreaLights.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanAreaLightsFragment {"AreaLights.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanAreaLightsWeighted {"AreaLights.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_TRANSPARENCY=1"};

inline constexpr VulkanShaderProgram VulkanTechniqueShaders[][2] = {
    {VulkanPhongVertex, VulkanPhongFragment},
    {VulkanFlatVertex, VulkanFlatFragment},
    {VulkanGouraudVertex, VulkanGouraudFragment},
    {VulkanPBRVertex, VulkanPBRFragment},
    {VulkanToonVertex, VulkanToonFragment},
    {VulkanAreaLightsVertex, VulkanAreaLightsFragment},
};

// VolumetricFog source variants.
inline constexpr VulkanShaderProgram VulkanVolumetricFogVertex {"VolumetricFog.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanVolumetricFogFragment {"VolumetricFog.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanVolumetricFogShaders[2] = {
    VulkanVolumetricFogVertex,
    VulkanVolumetricFogFragment,
};

// Participating media compiles independent GLSL directly at runtime.
inline constexpr VulkanShaderProgram VulkanMediaTransportShaders[2] = {
    {"ParticipatingMedia.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"},
    {"ParticipatingMedia.glsl", VulkanShaderStage::Fragment, ""},
};
inline constexpr VulkanShaderProgram VulkanMediaCompositionShaders[2] = {
    {"ParticipatingMedia.glsl", VulkanShaderStage::Vertex, "EVANS_MEDIA_COMPOSITE=1;EVANS_VERTEX_STAGE=1"},
    {"ParticipatingMedia.glsl", VulkanShaderStage::Fragment, "EVANS_MEDIA_COMPOSITE=1"},
};

// Clouds retain their own GLSL producer and composition implementation.
inline constexpr VulkanShaderProgram VulkanCloudTransportShaders[2] = {
    {"VolumetricClouds.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"},
    {"VolumetricClouds.glsl", VulkanShaderStage::Fragment, ""},
};
inline constexpr VulkanShaderProgram VulkanCloudCompositionShaders[2] = {
    {"VolumetricClouds.glsl", VulkanShaderStage::Vertex, "EVANS_CLOUD_COMPOSITE=1;EVANS_VERTEX_STAGE=1"},
    {"VolumetricClouds.glsl", VulkanShaderStage::Fragment, "EVANS_CLOUD_COMPOSITE=1"},
};

// VolumetricLighting source variants.
inline constexpr VulkanShaderProgram VulkanVolumetricLightingVertex {"VolumetricLighting.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanVolumetricLightingFragment {"VolumetricLighting.glsl", VulkanShaderStage::Fragment, ""};

inline constexpr VulkanShaderProgram VulkanVolumetricLightingShaders[2] = {
    VulkanVolumetricLightingVertex,
    VulkanVolumetricLightingFragment,
};

// VSM source variants.
inline constexpr VulkanShaderProgram VulkanVSMVertex {"VSM.glsl", VulkanShaderStage::Vertex, "EVANS_VSM_PROGRAM=5;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanVSMCasterVertex {"VSM.glsl", VulkanShaderStage::Vertex, "EVANS_VSM_PROGRAM=8;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanVSMInstanceVertex {"VSM.glsl", VulkanShaderStage::Vertex, "EVANS_VSM_PROGRAM=8;EVANS_VERTEX_STAGE=1;EVANS_VSM_INSTANCED=1"};
inline constexpr VulkanShaderProgram VulkanVSMFilterVertex {"VSM.glsl", VulkanShaderStage::Vertex, "EVANS_VSM_PROGRAM=9;EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanVSMFragment0 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=0"};
inline constexpr VulkanShaderProgram VulkanVSMFragment1 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=1"};
inline constexpr VulkanShaderProgram VulkanVSMFragment2 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=2"};
inline constexpr VulkanShaderProgram VulkanVSMFragment3 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=3"};
inline constexpr VulkanShaderProgram VulkanVSMFragment4 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=4"};
inline constexpr VulkanShaderProgram VulkanVSMFragment5 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=5"};
inline constexpr VulkanShaderProgram VulkanVSMFragment6 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=6"};
inline constexpr VulkanShaderProgram VulkanVSMFragment7 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=7"};
inline constexpr VulkanShaderProgram VulkanVSMFragment8 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=8"};
inline constexpr VulkanShaderProgram VulkanVSMFragment9 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=9"};
inline constexpr VulkanShaderProgram VulkanVSMFragment10 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=0;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanVSMFragment11 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=1;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanVSMFragment12 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=2;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanVSMFragment13 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=3;EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanVSMFragment14 {"VSM.glsl", VulkanShaderStage::Fragment, "EVANS_VSM_PROGRAM=4;EVANS_WEIGHTED_TRANSPARENCY=1"};

inline constexpr VulkanShaderProgram VulkanVSMFragments[15] = {
    VulkanVSMFragment0,
    VulkanVSMFragment1,
    VulkanVSMFragment2,
    VulkanVSMFragment3,
    VulkanVSMFragment4,
    VulkanVSMFragment5,
    VulkanVSMFragment6,
    VulkanVSMFragment7,
    VulkanVSMFragment8,
    VulkanVSMFragment9,
    VulkanVSMFragment10,
    VulkanVSMFragment11,
    VulkanVSMFragment12,
    VulkanVSMFragment13,
    VulkanVSMFragment14,
};
inline constexpr VulkanShaderProgram VulkanVSMScreenShaders[3][2] = {
    {VulkanVSMVertex, VulkanVSMFragments[5]},
    {VulkanVSMVertex, VulkanVSMFragments[6]},
    {VulkanVSMVertex, VulkanVSMFragments[7]},
};
inline constexpr VulkanShaderProgram VulkanVSMCasterShaders[2][2] = {
    {VulkanVSMCasterVertex, VulkanVSMFragments[8]},
    {VulkanVSMInstanceVertex, VulkanVSMFragments[8]},
};
inline constexpr VulkanShaderProgram VulkanVSMFilterShaders[2] = {
    VulkanVSMFilterVertex, VulkanVSMFragments[9]
};

// WeightedTransparency source variants.
inline constexpr VulkanShaderProgram VulkanWeightedPhong {"Phong.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanWeightedFlat {"Flat.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanWeightedGouraud {"Gouraud.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanWeightedPBR {"PBR.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanWeightedToon {"Toon.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_TRANSPARENCY=1"};
inline constexpr VulkanShaderProgram VulkanWeightedSingleVertex {"WeightedBlendedOIT.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1"};
inline constexpr VulkanShaderProgram VulkanWeightedSingleFragment {"WeightedBlendedOIT.glsl", VulkanShaderStage::Fragment, ""};
inline constexpr VulkanShaderProgram VulkanWeightedSamplesVertex {"WeightedBlendedOIT.glsl", VulkanShaderStage::Vertex, "EVANS_VERTEX_STAGE=1;EVANS_WEIGHTED_SAMPLES=1"};
inline constexpr VulkanShaderProgram VulkanWeightedSamplesFragment {"WeightedBlendedOIT.glsl", VulkanShaderStage::Fragment, "EVANS_WEIGHTED_SAMPLES=1"};

inline constexpr VulkanShaderProgram VulkanWeightedSurfaceShaders[] = {
    VulkanWeightedPhong,
    VulkanWeightedFlat,
    VulkanWeightedGouraud,
    VulkanWeightedPBR,
    VulkanWeightedToon,
    VulkanAreaLightsWeighted,
};
inline constexpr VulkanShaderProgram VulkanWeightedResolveShaders[2][2] = {
    {VulkanWeightedSingleVertex, VulkanWeightedSingleFragment},
    {VulkanWeightedSamplesVertex, VulkanWeightedSamplesFragment},
};

}
