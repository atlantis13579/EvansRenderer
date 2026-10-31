# Evans

Independent C++17 real-time renderer and software rasterizer. Supported rendering
backends are DirectX 11, DirectX 12, DXR, Vulkan, OpenGL ES, Metal, and software
rasterization on CPU, CUDA, Compute Shader, or Metal. Unsupported native devices
are omitted from the UI. Scenes: Triangle, Utah Teapot, and Sponza Atrium.

## Layout

- `Src/`: C++/Objective-C++ source, shaders, and validation.
- `Contents/`: independently owned models and textures; see `Contents/ATTRIBUTION.md`.
- `Build/`: build definitions and generated VS 2022/Xcode projects.
- `Intermediate/`: compiler objects, libraries, symbols, and validation captures (ignored).
- `Binaries/`: final executables by configuration (ignored).

## Windows / Visual Studio 2022

```powershell
.\Build\GenerateVS2022.bat
cmake --build Build/VS2022 --config Release --parallel
.\Binaries\Release\EvansEditor.exe
ctest --test-dir Build/VS2022 -C Release --output-on-failure
```

Open `Build/VS2022/Evans.sln`. MSBuild directory targets route compiler output to
`Intermediate/VS2022`; final executables are in `Binaries/Release`.
The editor locates `Contents/` from the executable or working directory, so the
project can be moved without embedding its build path in the executable.

## macOS / Xcode

```sh
bash Build/GenerateXcode.sh
cmake --build Build/Xcode --config Release
Binaries/Release/EvansEditor
ctest --test-dir Build/Xcode -C Release --output-on-failure
```

Xcode is generated on macOS. Metal rendering and presentation are selected there.

## Controls and validation

The ImGui panel selects the backend, scene, fill mode, shadows, and outline.
WASD and right mouse control the camera; click geometry to select it. Software
rasterization updates automatically when the camera, scene, or settings change.

`evans.renderer_matrix` covers every available backend with all three scenes,
fill/shadow/outline combinations, frame parity within each rendering family,
resize/settings round trips, directed backend switches, and camera/animation
invalidation. Native presentation tests exercise the window message loop and
backend switching. Tests may take several minutes and write ignored captures to
`Intermediate/Validation`.

Regenerate embedded shader headers after changing their sources:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Build/GenerateDxrShaders.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File Build/GenerateRasterShaders.ps1
```

The DXR script requires DXC; the CUDA raster script requires nvcc and MSVC.
Embedded shader headers are runtime inputs and are intentionally tracked.

## Verified configuration

On 2026-10-30, the VS 2022 Release build passed all 13 registered Windows tests
on an NVIDIA GeForce RTX 4060 (driver 617.14). The renderer matrix exercised
8 available backends, 288 effect frames, 56 directed switches, and 180 family
image comparisons. Native-window tests compared 2,016 frames across Triangle,
Utah Teapot, and Sponza, including animated switching, rapid selections, and
minimize/restore cycles. CPU, CUDA, and Compute rasterizers also passed automatic
animation tests without a manual render request. Both shader-generation scripts
reproduced the checked-in headers. Xcode and native Metal execution require a
Mac and were not run during this Windows validation.
