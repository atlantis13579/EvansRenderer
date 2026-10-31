cmake_minimum_required(VERSION 3.20)

project(Evans VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

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

function(evans_target target folder)
    target_include_directories(${target} PUBLIC "${EVANS_ROOT}/Src")
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES FOLDER "${folder}")
    if(MSVC)
        set_target_properties(${target} PROPERTIES VS_GLOBAL_IntDir "${EVANS_ROOT}/Intermediate/VS2022/$(ProjectName)/$(Configuration)/")
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(EvansMaths STATIC
    "${EVANS_ROOT}/Src/Maths/StaticAABBTree.cpp"
    "${EVANS_ROOT}/Src/Maths/TriMesh.cpp"
)
evans_target(EvansMaths Modules)

add_library(EvansRenderer STATIC
    "${EVANS_ROOT}/Src/Renderer/DX11RHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/DX12RHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/DxrRayTracingRHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/MetalRHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/OpenGLESRHI.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracing.cpp"
    "${EVANS_ROOT}/Src/Renderer/RayTracingBackendCommon.cpp"
    "${EVANS_ROOT}/Src/Renderer/RealtimeShader.cpp"
    "${EVANS_ROOT}/Src/Renderer/Renderer.cpp"
    "${EVANS_ROOT}/Src/Renderer/RHIBackends.cpp"
    "${EVANS_ROOT}/Src/Renderer/RHIThread.cpp"
    "${EVANS_ROOT}/Src/Renderer/VulkanRHI.cpp"
)
evans_target(EvansRenderer Modules)
target_compile_definitions(EvansRenderer PUBLIC EVANS_RENDERER_STATIC)
target_link_libraries(EvansRenderer PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(EvansRenderer PUBLIC d3d11 d3d12 dxgi d3dcompiler)
endif()

add_library(EvansRasterizer STATIC
    "${EVANS_ROOT}/Src/Rasterizer/BackendCommon.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/Backends.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/ComputeShaderRasterRHI.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/CpuRasterRHI.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/CudaRasterRHI.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/GpuRasterCommon.cpp"
    "${EVANS_METAL_RASTER_SOURCE}"
    "${EVANS_ROOT}/Src/Rasterizer/Rasterizer.cpp"
    "${EVANS_ROOT}/Src/Rasterizer/SoftwareRasterCore.cpp"
)
evans_target(EvansRasterizer Modules)
target_link_libraries(EvansRasterizer PUBLIC Threads::Threads)
if(APPLE)
    target_link_libraries(EvansRasterizer PUBLIC
        "-framework Foundation"
        "-framework Metal"
    )
endif()

add_library(EvansEditorCore STATIC
    "${EVANS_ROOT}/Src/Editor/ContentRoot.cpp"
    "${EVANS_ROOT}/Src/Editor/Demos.cpp"
    "${EVANS_ROOT}/Src/Editor/Editor.cpp"
    "${EVANS_ROOT}/Src/Editor/ModuleAdapters.cpp"
    "${EVANS_ROOT}/Src/Editor/RenderAdapter.cpp"
    "${EVANS_ROOT}/Src/Editor/Resources.cpp"
)
evans_target(EvansEditorCore Modules)
target_link_libraries(EvansEditorCore PUBLIC
    Threads::Threads
    EvansMaths
    EvansRenderer
    EvansRasterizer
)
if(WIN32)
    target_link_libraries(EvansEditorCore PUBLIC ole32 windowscodecs)
elseif(APPLE)
    target_link_libraries(EvansEditorCore PUBLIC
        "-framework CoreGraphics"
        "-framework ImageIO"
    )
endif()

if(WIN32)
    add_executable(EvansEditor WIN32
        "${EVANS_ROOT}/Src/Editor/EditorPresenter.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterDX11.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterDX12.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterOpenGLES.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterVulkan.cpp"
        "${EVANS_ROOT}/Src/Editor/Imgui.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderDX11.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderDX12.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderOpenGLES.cpp"
        "${EVANS_ROOT}/Src/Editor/ImguiRenderVulkan.cpp"
        "${EVANS_ROOT}/Src/Editor/Main.cpp"
    )
    target_link_libraries(EvansEditor PRIVATE
        EvansEditorCore
        d3d11
        d3d12
        d3dcompiler
        dxgi
        gdi32
        shell32
        user32
    )
    target_compile_definitions(EvansEditor PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    add_executable(EvansEditor
        "${EVANS_ROOT}/Src/Editor/EditorPresenter.cpp"
        "${EVANS_ROOT}/Src/Editor/EditorPresenterMetal.mm"
        "${EVANS_ROOT}/Src/Editor/Imgui.cpp"
        "${EVANS_ROOT}/Src/Editor/Main.cpp"
        "${EVANS_ROOT}/Src/Editor/MainMac.mm"
    )
    target_link_libraries(EvansEditor PRIVATE
        EvansEditorCore
        "-framework Cocoa"
        "-framework CoreGraphics"
        "-framework Metal"
        "-framework QuartzCore"
    )
else()
    add_executable(EvansEditor "${EVANS_ROOT}/Src/Editor/Main.cpp")
    target_link_libraries(EvansEditor PRIVATE EvansEditorCore)
endif()
evans_target(EvansEditor Editor)

if(BUILD_TESTING)
    add_executable(EvansRendererMatrix "${EVANS_ROOT}/Src/Validation/RendererMatrix.cpp")
    evans_target(EvansRendererMatrix Validation)
    target_link_libraries(EvansRendererMatrix PRIVATE EvansEditorCore)
    add_test(NAME evans.renderer_matrix COMMAND $<TARGET_FILE:EvansRendererMatrix>)
    set_tests_properties(evans.renderer_matrix PROPERTIES TIMEOUT 600 RUN_SERIAL TRUE)

    set(EVANS_TEST_OUTPUT_DIRECTORY "${EVANS_ROOT}/Intermediate/Validation")
    file(MAKE_DIRECTORY "${EVANS_TEST_OUTPUT_DIRECTORY}")

    add_test(
        NAME evans.catalog
        COMMAND $<TARGET_FILE:EvansEditor> --catalog-smoke)
    add_test(
        NAME evans.shader_parser
        COMMAND $<TARGET_FILE:EvansEditor> --shader-parser-smoke)
    add_test(
        NAME evans.switch
        COMMAND $<TARGET_FILE:EvansEditor>
            --switch-smoke
            --output "${EVANS_TEST_OUTPUT_DIRECTORY}/switch.ppm")
    add_test(
        NAME evans.switch_stress
        COMMAND $<TARGET_FILE:EvansEditor>
            --switch-stress-smoke
            --output "${EVANS_TEST_OUTPUT_DIRECTORY}/switch-stress.ppm")

    set_tests_properties(
        evans.catalog
        evans.shader_parser
        PROPERTIES TIMEOUT 30)
    set_tests_properties(
        evans.switch
        evans.switch_stress
        PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)

    set(EVANS_ANIMATED_SOFTWARE_BACKENDS cpu)
    if(WIN32)
        list(APPEND EVANS_ANIMATED_SOFTWARE_BACKENDS cuda compute)
    elseif(APPLE)
        list(APPEND EVANS_ANIMATED_SOFTWARE_BACKENDS metal)
    endif()
    foreach(backend IN LISTS EVANS_ANIMATED_SOFTWARE_BACKENDS)
        add_test(NAME evans.animation_${backend}
            COMMAND $<TARGET_FILE:EvansEditor> --realtime-smoke
                --backend rasterizer.${backend}
                --output "${EVANS_TEST_OUTPUT_DIRECTORY}/animation-${backend}.ppm")
        set_tests_properties(evans.animation_${backend} PROPERTIES
            TIMEOUT 180 SKIP_RETURN_CODE 4 RUN_SERIAL TRUE)
    endforeach()

    if(WIN32)
        add_test(
            NAME evans.presentation_pair
            COMMAND $<TARGET_FILE:EvansEditor> --presentation-pair-smoke
                --demo sponza
                --output "${EVANS_TEST_OUTPUT_DIRECTORY}/presentation-pair.txt")
        set_tests_properties(evans.presentation_pair PROPERTIES
            TIMEOUT 300 RUN_SERIAL TRUE)
        foreach(EVANS_LIVE_DEMO triangle teapot sponza)
            add_test(
                NAME evans.presentation_live_${EVANS_LIVE_DEMO}
                COMMAND $<TARGET_FILE:EvansEditor> --presentation-live-smoke
                    --demo ${EVANS_LIVE_DEMO}
                    --output "${EVANS_TEST_OUTPUT_DIRECTORY}/presentation-live-${EVANS_LIVE_DEMO}.txt")
            set_tests_properties(evans.presentation_live_${EVANS_LIVE_DEMO}
                PROPERTIES TIMEOUT 2400 RUN_SERIAL TRUE)
        endforeach()
        add_test(
            NAME evans.presentation_switch
            COMMAND $<TARGET_FILE:EvansEditor> --presentation-switch-smoke)
        set_tests_properties(
            evans.presentation_switch
            PROPERTIES
                TIMEOUT 180
                SKIP_RETURN_CODE 4
                RUN_SERIAL TRUE)
    endif()
endif()

if(MSVC)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT EvansEditor)
endif()
