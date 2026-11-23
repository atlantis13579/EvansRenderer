# Evans

Evans is an independent C++17 renderer for studying graphics algorithms on shared
scenes. Native raster backends cover DirectX 11, DirectX 12, Vulkan, OpenGL ES, and
Metal. Software rasterizers run on CPU, CUDA, DirectX compute, and Metal compute. DXR
demonstrates hardware ray visibility and direct lighting.

CUDA and DirectX compute kernels use global memory to transform and clip triangles,
build tile lists, test coverage and depth, filter texels, and shade pixels. They run
independently of hardware rasterization, depth testing, texture samplers, and ray
tracing units. The optional Windows bridge publishes the finished color buffer as a
shared image for display.

Native raster APIs use readable, independently authored, self-contained technique
shaders. Shared C++ scene contracts define resources and data layouts. Frustum culling
and mesh LOD run on the CPU. Back-face culling uses API state. Algorithms, numerical
parameters, and research references accompany their implementations.

## Demo scenes

Triangle validates the backend. The rotating Utah Teapot compares texture mapping,
shadows, and environment reflections.

<img width="1280" height="720" alt="Utah Teapot" src="https://github.com/user-attachments/assets/1d7ecd69-586b-4c46-a0f0-b43a3ffab38b" />



Fur Cube shows procedural strands and hair shading.

<img width="1280" height="720" alt="Hair Rendering" src="https://github.com/user-attachments/assets/af7dd096-7f77-4a9d-8402-82ef1051f33e" />



Sponza Atrium shows color, depth, SSAO, and normal buffers.

<img width="1280" height="720" alt="Sponza Atrium" src="https://github.com/user-attachments/assets/6cf96da7-f70f-4a62-b43f-fa4a7c95f5c1" />



Four Stanford Dragon exhibits show jade diffusion, glass refraction, alpha transparency,
and planar reflection.


<img width="1280" height="720" alt="Stanford Dragon" src="https://github.com/user-attachments/assets/b3ffaaa8-80b9-4580-bd18-3b8e208ee0b6" />

Big World combines procedural mountains, forests, and a lake with day-night lighting,
rain, snow, fog, and real-time volumetric clouds. Lake waves combine an FFT spectrum
with Gerstner swells.



<img width="1280" height="5892" alt="Environment Changes" src="https://github.com/user-attachments/assets/06c9c2e5-1754-4f52-8f1d-431eda18ad4b" />



## Rendering techniques

These techniques run on all five native raster APIs unless noted. Software rasterization
and DXR have smaller capability sets.

| Family | Algorithms |
| --- | --- |
| Pipelines | Forward, five-target Deferred, Forward+, Clustered Forward, Tiled Deferred, Visibility Buffer, hybrid raster/ray rendering, progressive Path Tracing |
| Shading | Flat geometric normals, Gouraud vertex lighting, per-fragment Phong, GGX metallic/roughness PBR, Toon bands, image-space NPR, screen-space subsurface diffusion, procedural hair/fur, rectangle area-light quadrature |
| Materials | Existing OBJ/MTL materials, texture and tangent-space normal mapping, alpha testing, sorted alpha blending, parallax occlusion mapping, dielectric refraction |
| Shadows | Single maps, stable cascades, fixed-radius PCF (basic soft shadows), contact-hardening PCSS, filtered variance moments, demand-paged Virtual Shadow Maps, screen-space contact shadows, scene-BVH ray shadows |
| Ambient occlusion | SSAO, horizon-based HBAO, mip-based SAO, spatial GTAO, scene-BVH ray-traced AO |
| Diffuse global illumination | One-bounce baked GI (Baked using [ArvoRenderer](https://github.com/atlantis13579/ArvoRenderer)), generated UV2 lightmaps, visibility-weighted L2 SH probes, temporal SSGI, directional voxel cone tracing, bounded scene-BVH diffuse paths |
| Reflections | Runtime cubemap and latitude-longitude captures, mirrored planar captures, SSR, GGX scene-BVH reflections. Hybrid rendering combines screen intersections with scene-ray fallback. |
| Geometry | Frustum and back-face culling, quadric mesh LOD, indexed instancing, GPU instance compaction and indirect drawing, compute PN triangle tessellation, geometric displacement, streamed virtual geometry |
| Antialiasing | Native MSAA with explicit resolves, display-domain FXAA, motion-reprojected TAA, independent temporal super resolution (TSR) |
| Volumes | Height fog, shadowed volumetric lighting, procedural heterogeneous participating media, procedural volumetric clouds, screen-space light shafts |
| Post-processing | Reinhard and ACES-fitted tone mapping, bloom, analytic color grading, thin-lens depth of field, rigid motion blur, screen-space distortion, lens flare, film grain, vignette, temporal exposure adaptation |
| Transparency | Sorted source-over blending, weighted blended OIT, bounded depth-peeling OIT |


## Experimental rendering

Lumen, Nanite, mesh shading, and Gaussian Splatting are standalone teaching
implementations with limited scope. They run independently of the Unreal Engine
runtime and plugins.

- **Lumen** uses mesh and global distance fields, material cards, surface-cache
  radiosity, hierarchical screen tracing, screen probes, and scrolling world-radiance
  reuse. Diffuse GI and GGX reflections have separate reconstruction and histories, and
  a frustum SH volume lights translucency. It uses one adaptive screen-probe subdivision
  level, finite angular budgets, constant sky radiance, and single-sample material
  capture. The world cache uses finite-radius parallax with one depth layer and no
  angular mip filtering. The coarse translucent volume can leak through thin surfaces.
- **Nanite** streams cluster-group DAGs and maintains complete resident geometry cuts.
  The GPU selects and decodes clusters, combines hardware and compute visibility, and
  resolves materials. Host material and family binding remain in place, with finite
  streaming, page, and scratch budgets.
- **Mesh shaders** use native DX12 and Vulkan mesh dispatch or Metal object/mesh stages
  on supported devices. They accept ordinary visibility triangles or Nanite hardware
  bins. DX11 and OpenGL ES lack native mesh stages.
- **Gaussian Splatting** represents a scene as translucent 3D Gaussians.
  Their positions, anisotropic scales, rotations, opacity, and degree-zero color
  are trained using [ElmanLLM](https://github.com/atlantis13579/Elman)
  framework. Evans projects them to screen-space ellipses, sorts them by camera
  distance, and alpha-composites the splats into the final image.
