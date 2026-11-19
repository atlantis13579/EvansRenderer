# DXC compiles readable mesh HLSL on demand. Deploy the existing Windows SDK
# compiler/runtime beside the application; no shader bytecode is generated.
set(EVANS_DXC_RUNTIME_DIR "" CACHE PATH "Directory containing the runtime DXC DLLs")
file(TO_CMAKE_PATH "$ENV{ProgramFiles\(x86\)}" EVANS_PROGRAM_FILES_X86)
file(GLOB EVANS_DXC_SDK_DIRECTORIES LIST_DIRECTORIES true
    "${EVANS_PROGRAM_FILES_X86}/Windows Kits/10/bin/*/x64")
list(SORT EVANS_DXC_SDK_DIRECTORIES COMPARE NATURAL ORDER DESCENDING)
find_file(EVANS_DXCOMPILER_DLL dxcompiler.dll HINTS "${EVANS_DXC_RUNTIME_DIR}" ${EVANS_DXC_SDK_DIRECTORIES})
find_file(EVANS_DXIL_DLL dxil.dll HINTS "${EVANS_DXC_RUNTIME_DIR}" ${EVANS_DXC_SDK_DIRECTORIES})
if(EVANS_DXCOMPILER_DLL AND EVANS_DXIL_DLL)
    add_custom_command(TARGET EvansEditor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${EVANS_DXCOMPILER_DLL}" "$<TARGET_FILE_DIR:EvansEditor>/dxcompiler.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${EVANS_DXIL_DLL}" "$<TARGET_FILE_DIR:EvansEditor>/dxil.dll"
        VERBATIM)
endif()
