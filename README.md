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
