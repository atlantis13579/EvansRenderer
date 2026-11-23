cmake_minimum_required(VERSION 3.22.1)

project(Evans VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(APPLE)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT EvansEditor)

get_filename_component(EVANS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(APPLE)
    set(EVANS_METAL_RASTER_SOURCE
        "${EVANS_ROOT}/Src/Rasterizer/MetalRasterRHI.mm")
else()
    set(EVANS_METAL_RASTER_SOURCE
        "${EVANS_ROOT}/Src/Rasterizer/MetalRasterRHIStub.cpp")
endif()
find_package(Threads REQUIRED)
set(CMAKE_XCODE_ATTRIBUTE_OBJROOT "${EVANS_ROOT}/Intermediate/Xcode")
set(CMAKE_XCODE_ATTRIBUTE_SYMROOT "${EVANS_ROOT}/Intermediate/Xcode")
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY "${EVANS_ROOT}/Intermediate/PDB")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${EVANS_ROOT}/Intermediate/PDB")

foreach(configuration Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${configuration}" configuration_upper)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${configuration_upper} "${EVANS_ROOT}/Intermediate/Lib/${configuration}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${configuration_upper} "${EVANS_ROOT}/Intermediate/Lib/${configuration}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${configuration_upper} "${EVANS_ROOT}/Binaries/${configuration}")
endforeach()

add_library(EvansNeuralCore STATIC
    "${EVANS_ROOT}/Src/Renderer/Neural/Json.cpp"
    "${EVANS_ROOT}/Src/Renderer/Neural/GaussianScene.cpp"
    "${EVANS_ROOT}/Src/Renderer/Neural/GaussianGlb.cpp"
    "${EVANS_ROOT}/Src/Renderer/Neural/TrainingGraph.cpp"
    "${EVANS_ROOT}/Src/Renderer/Neural/TrainingData.cpp")
target_include_directories(EvansNeuralCore PUBLIC "${EVANS_ROOT}/Src")

add_executable(EvansEditor
    "${EVANS_ROOT}/Src/Maths/StaticAABBTree.cpp"
    "${EVANS_ROOT}/Src/Maths/TriMesh.cpp"

    "${EVANS_ROOT}/Src/Techniques/TechniqueCatalog.cpp"

    "${EVANS_ROOT}/Src/Renderer/ShaderCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/D3DShaderCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/OpenGLES/OpenGLESProgramCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12/DX12PipelineCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12/DXILShaderCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanPipelineCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Shadows/CascadedShadowMap.cpp"
    "${EVANS_ROOT}/Src/Renderer/Shadows/ShadowProjection.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX11/DX11RHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX11/DX11Multisampling.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX11/DX11ForwardPlus.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX11/DX11ClusteredForward.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12/DX12RHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12/DX12ForwardPlus.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12/DX12ClusteredForward.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12/DxrRayTracingRHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/FrustumCulling.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lighting/ForwardPlus.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lighting/ClusteredForward.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lighting/SceneLightCollection.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lighting/AreaLightSampling.cpp"
    "${EVANS_ROOT}/Src/Renderer/Transparency/DepthPeeling.cpp"
    "${EVANS_ROOT}/Src/Renderer/Transparency/RefractionGeometry.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/LevelOfDetail.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/MeshGeometry.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing/SceneRayQuery.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing/SceneRayUpload.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing/RayTracedShadowScene.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing/SceneRayShadingUpload.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/BakedLightProbes.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/BakedDiffuseLighting.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/ArvoLightmap.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/BakedLightmapCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/BakedLightingAssets.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/LightmapReceivers.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/BakedProbeCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/BakedLighting/LightmapAtlas.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/Instancing.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/GPUDrivenRendering.cpp"
    "${EVANS_ROOT}/Src/Renderer/Pipelines/VisibilityBuffer.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lighting/SubsurfaceDiffusion.cpp"
    "${EVANS_ROOT}/Src/Renderer/VoxelGI/VoxelRadiance.cpp"
    "${EVANS_ROOT}/Src/Renderer/VoxelGI/VoxelRadianceCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenMeshCards.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenMeshPartition.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenMeshRepresentation.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenHair.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenHairReceiver.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenHardwareRayTracing.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenScene.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenSceneCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenRenderCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenSurfaceCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenSurfaceLookup.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenSurfaceLighting.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenRadianceTrace.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenReflections.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenProbeSampling.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenEnvironment.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenProbeIntegration.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenTranslucency.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenProbeGather.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenTemporal.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenWorldRadianceCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenDistanceField.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenDistanceScene.cpp"
    "${EVANS_ROOT}/Src/Renderer/Lumen/LumenGlobalDistance.cpp"
    "${EVANS_ROOT}/Src/Renderer/Hair/HairGeometryCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Hair/HairStrands.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/HeightField.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/GeometryPreparation.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/TriangleSubdivision.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/DisplacementGeometry.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/PNTessellation.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualGeometry/VirtualGeometryPages.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualGeometry/VirtualGeometryResidency.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualGeometry/VirtualGeometryHierarchy.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteHierarchy.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NanitePages.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteAsset.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteSceneBudget.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteStreaming.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteGpu.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteStream.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteVisibility.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteRaster.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteFrameVisibility.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteDraws.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteMaterials.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteInstances.cpp"
    "${EVANS_ROOT}/Src/Renderer/Nanite/NaniteHardware.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/GraphPartition.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualGeometry/VirtualGeometryGpu.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualGeometry/VirtualGeometryCache.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualGeometry/VirtualGeometryStreaming.cpp"
    "${EVANS_ROOT}/Src/Renderer/PostProcessing/TemporalSuperResolution.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualShadowMaps/VirtualShadowMaps.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualShadowMaps/VirtualShadowScene.cpp"
    "${EVANS_ROOT}/Src/Renderer/Geometry/ParallaxOcclusionMapping.cpp"
    "${EVANS_ROOT}/Src/Renderer/Hair/HairScattering.cpp"
    "${EVANS_ROOT}/Src/Renderer/Hair/HairLighting.cpp"
    "${EVANS_ROOT}/Src/Renderer/Hair/HairShadowReceiver.cpp"
    "${EVANS_ROOT}/Src/Renderer/Metal/MetalRHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/OpenGLES/OpenGLESRHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/OpenGLES/OpenGLESContactShadows.cpp"
    "${EVANS_ROOT}/Src/Renderer/OpenGLES/OpenGLESForwardPlus.cpp"
    "${EVANS_ROOT}/Src/Renderer/OpenGLES/OpenGLESClusteredForward.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing/RayTracing.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing/RayTracingBackendCommon.cpp"
    "${EVANS_ROOT}/Src/Renderer/RealtimeShader.cpp"
    "${EVANS_ROOT}/Src/Renderer/Renderer.cpp"
    "${EVANS_ROOT}/Src/Renderer/VirtualShadowMaps/RendererVirtualShadows.cpp"
    "${EVANS_ROOT}/Src/Renderer/Reflections/EnvironmentCapture.cpp"
    "${EVANS_ROOT}/Src/Renderer/Reflections/EnvironmentRadiance.cpp"
    "${EVANS_ROOT}/Src/Renderer/Atmosphere/Atmosphere.cpp"
    "${EVANS_ROOT}/Src/Renderer/Volumetrics/DensityVolume.cpp"
    "${EVANS_ROOT}/Src/Renderer/Reflections/PlanarReflection.cpp"
    "${EVANS_ROOT}/Src/Renderer/RHIBackends.cpp"
    "${EVANS_ROOT}/Src/Renderer/RHIThread.cpp"
    "${EVANS_ROOT}/Src/Renderer/TechniqueShader.cpp"
    "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanRHI.cpp"

    "${EVANS_ROOT}/Src/Rasterizer/BackendCommon.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/Backends.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/ComputeShaderRasterRHI.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/D3D11GpuOutput.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/CpuRasterRHI.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/CudaRasterRHI.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/GpuRasterCommon.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/ParallelExecutor.cpp"
    "${EVANS_METAL_RASTER_SOURCE}"
    "${EVANS_ROOT}/Src/Rasterizer/Rasterizer.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/SoftwareRasterCore.cpp"

    "${EVANS_ROOT}/Src/Editor/AssetCooker.cpp"
    "${EVANS_ROOT}/Src/Editor/LightmapBake.cpp"
    "${EVANS_ROOT}/Src/Editor/DragonDemo.cpp"
    "${EVANS_ROOT}/Src/Editor/HairDemo.cpp"
    "${EVANS_ROOT}/Src/Editor/VolumeDemos.cpp"
    "${EVANS_ROOT}/Src/Editor/BigWorld.cpp"
    "${EVANS_ROOT}/Src/Editor/WaterWaves.cpp"
    "${EVANS_ROOT}/Src/Editor/SponzaDemo.cpp"
    "${EVANS_ROOT}/Src/Editor/ContentRoot.cpp"
    "${EVANS_ROOT}/Src/Editor/Demos.cpp"
    "${EVANS_ROOT}/Src/Editor/Editor.cpp"
    "${EVANS_ROOT}/Src/Editor/NeuralWorkspace.cpp"
    "${EVANS_ROOT}/Src/Editor/ModuleAdapters.cpp"
    "${EVANS_ROOT}/Src/Editor/RenderAdapter.cpp"
    "${EVANS_ROOT}/Src/Editor/Resources.cpp"

    "${EVANS_ROOT}/Src/Editor/Main.cpp"
)

target_include_directories(EvansEditor PRIVATE "${EVANS_ROOT}/Src")
target_compile_features(EvansEditor PRIVATE cxx_std_17)
target_compile_definitions(EvansEditor PRIVATE
    EVANS_RENDERER_STATIC
    EVANS_NANITE_CACHE_DIRECTORY="${EVANS_ROOT}/Intermediate/Nanite"
    EVANS_SHADER_SOURCE_DIRECTORY="${EVANS_ROOT}/Src/Shaders"
)
target_link_libraries(EvansEditor PRIVATE Threads::Threads EvansNeuralCore)
if(MSVC)
    target_compile_options(EvansEditor PRIVATE /W4 /permissive-)
else()
    target_compile_options(EvansEditor PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32 OR APPLE)
    target_sources(EvansEditor PRIVATE
        "${EVANS_ROOT}/Src/Editor/EditorPresenter.cpp"
        "${EVANS_ROOT}/Src/Editor/Imgui.cpp"
        "${EVANS_ROOT}/Src/Editor/TechniqueControls.cpp"
        "${EVANS_ROOT}/Src/Editor/NeuralPanel.cpp"
        "${EVANS_ROOT}/Src/Editor/LightmapPanel.cpp"
        "${EVANS_ROOT}/Src/Editor/HairControls.cpp"
        "${EVANS_ROOT}/Src/Editor/DragonControls.cpp"
        "${EVANS_ROOT}/Src/Editor/VolumeControls.cpp"
    )
endif()

if(WIN32)
    set_target_properties(EvansEditor PROPERTIES WIN32_EXECUTABLE TRUE)
    target_compile_definitions(EvansEditor PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_sources(EvansEditor PRIVATE
        "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanContactShadows.cpp"
        "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanForwardPlus.cpp"
        "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanClusteredForward.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterDX11.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterDX12.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterOpenGLES.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterVulkan.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderDX11.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderDX12.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderOpenGLES.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderVulkan.cpp"
    )
    target_link_libraries(EvansEditor PRIVATE d3d11 d3d12 d3dcompiler dxgi gdi32 ole32 shell32 user32 windowscodecs)
    include("${CMAKE_CURRENT_LIST_DIR}/VulkanRuntimeCompiler.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/DX12RuntimeCompiler.cmake")

    # Browser updates can remove the system ANGLE runtime. A supplied standalone
    # runtime keeps both the renderer and editor presenter independent of it.
    set(EVANS_ANGLE_RUNTIME_DIR "" CACHE PATH "Directory containing standalone ANGLE runtime DLLs")
    if(EVANS_ANGLE_RUNTIME_DIR)
        foreach(runtime libEGL.dll libGLESv2.dll)
            if(NOT EXISTS "${EVANS_ANGLE_RUNTIME_DIR}/${runtime}")
                message(FATAL_ERROR "Missing ANGLE runtime: ${EVANS_ANGLE_RUNTIME_DIR}/${runtime}")
            endif()
            add_custom_command(TARGET EvansEditor POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${EVANS_ANGLE_RUNTIME_DIR}/${runtime}"
                    "$<TARGET_FILE_DIR:EvansEditor>/${runtime}"
                VERBATIM
            )
        endforeach()
    endif()
elseif(APPLE)
    target_sources(EvansEditor PRIVATE
        "${EVANS_ROOT}/Src/Renderer/Metal/MetalShaderCache.mm"
        "${EVANS_ROOT}/Src/Renderer/Metal/MetalContactShadows.mm"
        "${EVANS_ROOT}/Src/Renderer/Metal/MetalForwardPlus.mm"
        "${EVANS_ROOT}/Src/Renderer/Metal/MetalClusteredForward.mm"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterMetal.mm"
        "${EVANS_ROOT}/Src/Editor/MainMac.mm"
    )
    target_link_libraries(EvansEditor PRIVATE
        "-framework Cocoa"
        "-framework CoreGraphics"
        "-framework Foundation"
        "-framework ImageIO"
        "-framework Metal"
        "-framework QuartzCore"
    )
endif()
