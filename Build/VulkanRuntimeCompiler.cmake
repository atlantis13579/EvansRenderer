# Build the compiler with the renderer's Debug/Release CRT and STL settings.
# Sources stay in the CMake build tree, so clearing Intermediate does not remove
# an external SDK. Shader compilation still happens on demand inside the editor.
include(FetchContent)

function(evans_configure_glslang)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()

    set(BUILD_SHARED_LIBS OFF)
    set(SPIRV_HEADERS_ENABLE_TESTS OFF)
    set(SPIRV_HEADERS_ENABLE_INSTALL OFF)
    set(SPIRV_SKIP_EXECUTABLES ON)
    set(SPIRV_SKIP_TESTS ON)
    set(SKIP_SPIRV_TOOLS_INSTALL ON)
    set(SPIRV_WERROR OFF)
    set(GLSLANG_TESTS OFF)
    set(GLSLANG_ENABLE_INSTALL OFF)
    set(ENABLE_GLSLANG_BINARIES OFF)
    set(ENABLE_HLSL OFF)
    set(ENABLE_EXCEPTIONS ON)
    set(BUILD_EXTERNAL OFF)

    # glslang 16.0.0 and the revisions in its known_good.json:
    # https://github.com/KhronosGroup/glslang/blob/16.0.0/known_good.json
    FetchContent_Declare(evans_spirv_headers
        URL https://codeload.github.com/KhronosGroup/SPIRV-Headers/tar.gz/01e0577914a75a2569c846778c2f93aa8e6feddd
        URL_HASH SHA256=494bd30dc13ba798af70edd989f8df82c90757c8ce9433598480f5e00e04c454
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps/downloads"
        DOWNLOAD_NAME spirv_headers.tar.gz
        TLS_VERIFY TRUE)
    FetchContent_MakeAvailable(evans_spirv_headers)
    set("SPIRV-Headers_SOURCE_DIR" "${evans_spirv_headers_SOURCE_DIR}")

    FetchContent_Declare(evans_spirv_tools
        URL https://codeload.github.com/KhronosGroup/SPIRV-Tools/tar.gz/7f2d9ee926f98fc77a3ed1e1e0f113b8c9c49458
        URL_HASH SHA256=0a92e4be5c0a60203048fcbfaa95061b1838038e62ccd57d7a6f17539d6d6daa
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps/downloads"
        DOWNLOAD_NAME spirv_tools.tar.gz
        TLS_VERIFY TRUE)
    FetchContent_MakeAvailable(evans_spirv_tools)

    FetchContent_Declare(evans_glslang
        URL https://codeload.github.com/KhronosGroup/glslang/tar.gz/a57276bf558f5cf94d3a9854ebdf5a2236849a5a
        URL_HASH SHA256=02f4321a3ed01b9a9a9874f76e2d8ad078825818fcc9aae12dbf2c01674fb8af
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps/downloads"
        DOWNLOAD_NAME glslang.tar.gz
        TLS_VERIFY TRUE)
    FetchContent_MakeAvailable(evans_glslang)

    # Upstream also defines optional libraries; only linked dependencies belong
    # in the default editor build, including the static SPIRV-Tools variant.
    foreach(source_directory IN ITEMS "${evans_spirv_headers_SOURCE_DIR}"
            "${evans_spirv_tools_SOURCE_DIR}" "${evans_glslang_SOURCE_DIR}")
        set_property(DIRECTORY "${source_directory}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    endforeach()
endfunction()

evans_configure_glslang()
target_sources(EvansEditor PRIVATE
    "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanGlslCompiler.cpp"
    "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanShaderCompiler.cpp")
target_compile_definitions(EvansEditor PRIVATE EVANS_SOURCE_DIRECTORY="${EVANS_ROOT}/Src")
# Invalidate persisted SPIR-V when compiler revisions, defaults or invocation
# flags change, including local edits to the native compiler implementation.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanGlslCompiler.cpp")
file(SHA256 "${CMAKE_CURRENT_LIST_FILE}" evans_vulkan_dependencies_hash)
file(SHA256 "${EVANS_ROOT}/Src/Renderer/Vulkan/VulkanGlslCompiler.cpp" evans_vulkan_invocation_hash)
target_compile_definitions(EvansEditor PRIVATE
    EVANS_VULKAN_COMPILER_ID="${evans_vulkan_dependencies_hash}-${evans_vulkan_invocation_hash}")
target_link_libraries(EvansEditor PRIVATE
    glslang::glslang glslang::glslang-default-resource-limits)
