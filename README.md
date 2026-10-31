# Evans

Independent C++17 real-time renderer and software rasterizer. Supported rendering
backends are DirectX 11, DirectX 12, DXR, Vulkan, OpenGL ES, Metal, and software
rasterization on CPU, CUDA, Compute Shader, or Metal. Unsupported native devices
are omitted from the UI. Scenes: Triangle, Utah Teapot, and Sponza Atrium.

## Rendering features

- Backend, scene, fill mode, shadows, and outline are selectable at runtime.
- Software rasterization updates automatically when the camera, scene, or settings change.
- Embedded shader headers are runtime inputs and are intentionally tracked.
- Models and textures under `Contents/` are independently owned; see `Contents/ATTRIBUTION.md`.

## Verified configuration

On 2026-10-30, the VS 2022 Release build passed all 13 registered Windows tests
on an NVIDIA GeForce RTX 4060 (driver 617.14). The renderer matrix exercised
8 available backends, 288 effect frames, 56 directed switches, and 180 family
image comparisons. Native-window tests compared 2,016 frames across Triangle,
Utah Teapot, and Sponza, including animated switching, rapid selections, and
minimize/restore cycles. CPU, CUDA, and Compute rasterizers also passed automatic
animation tests without a manual render request. Xcode and native Metal execution
require a Mac and were not run during this Windows validation.
