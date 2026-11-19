# Evans

Evans is an independent C++17 renderer for studying rendering algorithms on the
same scenes: Triangle, Utah Teapot, Sponza Atrium and Stanford Dragon. It combines
native raster rendering on DirectX 11, DirectX 12, Vulkan, OpenGL ES and Metal with
CPU, CUDA, DirectX compute and Metal compute software rasterizers. A separate DXR
backend demonstrates hardware ray visibility and direct lighting.

The CUDA and DirectX compute software rasterizers treat the GPU as parallel
workers with global memory. Kernels transform and clip triangles, build tile
lists, test coverage and depth, filter texels, and shade pixels. These steps
must not use hardware rasterization, depth testing, texture samplers, or ray
tracing units. On Windows the optional display bridge publishes the finished
color buffer as a shared image; it does not draw scene geometry.

The native raster APIs have independently authored, readable technique shaders.
A technique source owns its constituent passes and does not include another
technique or translate another API's source. Shared C++ scene contracts describe
resources and data layouts; they do not generate cross-API shaders. CPU algorithms
such as frustum culling and mesh LOD, and API states such as back-face culling,
do not require standalone shader programs. Algorithm explanations, numerical
parameters and research references live beside their C++ and shader implementations.

The catalog contains 84 entries: 78 implemented, three experimental (Lumen,
Nanite and Mesh Shaders), and three outside scope (DLSS, FSR and XeSS). The shader
inventory contains 85 common source units per raster API, including motion
vectors, plus three native mesh-stage sources: 428 files. Source coverage does
not imply device support or equal algorithm quality. Metal still requires native
compilation and execution on Apple hardware.

## Rendering techniques

Unless a narrower scope is stated, the following techniques have implementations
on all five native raster APIs. Software rasterization and DXR expose their own
smaller capability sets.

| Family | Implemented algorithms |
| --- | --- |
| Pipelines | Forward, five-target Deferred, Forward+, Clustered Forward, Tiled Deferred, Visibility Buffer, Hybrid raster/ray rendering and progressive Path Tracing. |
| Shading | Flat geometric normals, Gouraud vertex lighting, per-fragment Phong, GGX metallic/roughness PBR, Toon bands, image-space NPR, screen-space subsurface diffusion, procedural hair/fur and rectangle area-light quadrature. |
| Materials | Existing OBJ/MTL materials, texture and tangent-space normal mapping, alpha testing, sorted alpha blending, parallax occlusion mapping and dielectric refraction. |
| Shadows | Single shadow maps, stable cascades, fixed-radius PCF/basic soft shadows, contact-hardening PCSS, filtered variance moments, demand-paged Virtual Shadow Maps, screen-space contact shadows and scene-BVH ray shadows. |
| Ambient occlusion | SSAO, horizon-based HBAO, mip-based SAO, spatial GTAO and scene-BVH ray-traced AO. |
| Diffuse global illumination | One-bounce baked GI, generated UV2 lightmaps, visibility-weighted L2 SH probes, temporal SSGI, directional voxel cone tracing and bounded scene-BVH diffuse paths. |
| Reflections | Runtime cubemap and latitude-longitude environment captures, mirrored planar captures, SSR and GGX scene-BVH reflections. Hybrid rendering combines screen intersections with scene-ray fallback. |
| Geometry | Frustum and back-face culling, quadric mesh LOD, indexed instancing, GPU instance compaction/indirect drawing, compute PN triangle tessellation, geometric displacement and streamed virtual geometry. |
| Antialiasing | Native MSAA with explicit resolves, display-domain FXAA, motion-reprojected TAA and independent temporal super resolution. |
| Volumes | Height fog, shadowed volumetric lighting, procedural heterogeneous participating media, procedural volumetric clouds and screen-space light shafts. |
| Post-processing | Reinhard/ACES-fitted tone mapping, Bloom, analytic Color Grading, thin-lens Depth of Field, rigid Motion Blur, Screen-Space Distortion, Lens Flare, Film Grain, Vignette and temporal exposure adaptation. |
| Transparency | Sorted source-over blending, Weighted Blended OIT and bounded depth-peeling OIT. |

## Dependencies and composition

The editor retains requested settings and derives an effective configuration from
API support, device capabilities and resource dependencies. Inactive alternatives
remain available when the owning mode is disabled. The GUI uses the same
resolution rules as rendering and explains incompatible combinations.

| Selection | Relationship |
| --- | --- |
| Rendering pipeline | One active pipeline. Deferred-family pipelines reuse opaque material inputs; transparency remains forward shaded. Gouraud requires Forward. |
| Surface shading | Flat, Gouraud, Phong, PBR and Toon are exclusive. Flat and Gouraud suppress fragment normal mapping. NPR is a separate image effect. Physically based area lights require PBR and HDR. |
| Shadows | Single, cascaded, virtual and ray-traced coverage are alternatives. Hard, PCF, PCSS and VSM are alternative map filters. Contact shadows supplement primary visibility. Cascades require a directional light. |
| AO, diffuse GI and reflections | Each family selects one estimator. Different families compose. Baked lighting is an input producer; lightmaps and probes are alternative diffuse receivers. |
| Geometry | GPU-driven instance drawing requires instancing. Nanite replaces ordinary LOD and separate virtual geometry, and requires single-sample visibility rendering. Native mesh stages can consume ordinary visibility geometry or Nanite hardware bins. |
| Antialiasing | FXAA, TAA and TSR occupy one selector. MSAA is a separate geometry sample setting where supported; visibility rendering, depth peeling and nearest-layer refraction require a single-sample effective configuration. |
| Transparency | Sorted blending, Weighted Blended OIT and depth peeling are exclusive composition methods. Refraction uses opaque scene color/depth and nearest-layer data. |
| Temporal stages | Motion vectors, valid correspondence and private committed histories are produced as required. Camera cuts, resizing, topology and pipeline changes invalidate incompatible histories. |
| HDR stages | Floating scene-linear color is allocated for its consumers. Bloom, grading and flare precede tone mapping; display effects follow it. A required resolve supplies Reinhard when no tone operator was requested. |
| Path Tracing | Owns camera visibility, transport, antialiasing and progressive accumulation. Raster settings are retained for switching back. Display color changes preserve accumulated radiance. |

Captures, scene/cache preparation and primary shadows precede opaque visibility
and lighting. Resolved depth/materials feed AO, diffuse GI and reflections.
Temporal and lens effects operate on opaque HDR before sharp transparency.
HDR/display processing precedes selection outlines. The pass plan records input
producers and consumers; each RHI owns native resources and executes its passes.
History publication follows successful presentation, so failed frames cannot
replace the last committed correspondence or accumulation state.

Forward+ builds separate opaque-depth and full-frustum transparent tile lists.
Clustered Forward assigns lights to depth-sliced cells without requiring an opaque
depth prepass. Tiled Deferred combines G-buffer lighting with tile assignment.
Captures, volumetric samples and overflowing light lists retain complete lighting.
Visibility rendering stores triangle identity and reconstructs perspective-correct
material attributes before deferred lighting.

## Experimental rendering

Lumen, Nanite and mesh shading are independent teaching implementations with no
Unreal Engine runtime or plugin dependency and no claim of UE5 feature parity.

| Technique | Implemented principle and scope |
| --- | --- |
| Lumen | Mesh/global distance fields, material cards, surface-cache radiosity, hierarchical screen tracing, screen probes and scrolling world-radiance reuse. Diffuse GI and GGX reflections have separate reconstruction/history; a frustum SH volume supplies translucent lighting. |
| Nanite | Streamed cluster-group DAGs, complete resident geometry cuts, GPU selection/decoding, hybrid hardware/compute visibility and material resolve. Finite page and scratch budgets bound work. |
| Mesh Shaders | Native DX12/Vulkan mesh dispatch and Metal object/mesh stages, gated by device support. Works with ordinary visibility triangles or Nanite hardware bins. DX11 and OpenGL ES have no native mesh stage. |

Lumen uses one adaptive screen-probe subdivision level, finite angular budgets,
constant sky radiance and single-sample material capture. Its world cache uses
finite-radius parallax without angular mip filtering or multiple depth layers;
the coarse translucent volume can leak through thin surfaces. Nanite retains
host material/family binding and finite streaming budgets.

## Algorithm boundaries

Screen-space AO, GI, reflections and contact shadows have incomplete off-screen
information. Environment captures use ordinary mip filtering rather than GGX
prefiltering, and local probes retain parallax error. Scene-BVH raster effects use
shader traversal rather than requiring hardware ray-query support.

Path Tracing currently supports opaque/cutout meshes; transmission and
participating-media transport remain TODOs. Hair uses generated tapered ribbons;
scene-ray acceleration contains meshes rather than fiber primitives. Subsurface
diffusion operates on visible direct diffuse; thickness-based transmission remains
separate. Motion Blur handles rigid camera/object motion, and lens/temporal
reconstruction cannot recover hidden layers from a single depth image.

Current techniques reuse existing demo geometry/materials and generated render
targets, UV charts, probes, procedural height fields, strands and densities.
Additional authored inputs, such as measured fiber data, thickness or density
assets, remain explicit TODOs where required. No vendor upscaling SDK is included.
Models and textures retain their respective licenses.
