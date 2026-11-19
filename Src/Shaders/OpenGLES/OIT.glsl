#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Native ES 3.0 depth peeling: no atomics, storage buffers or shader includes.
// D32 depth remains [0,1]; only vertex clip depth converts to GL's [-1,1].
// Everitt (2001), Interactive Order-Independent Transparency; front compositing
// follows Bavoil/Myers (2008), section 2.2:
// https://developer.download.nvidia.com/SDK/10/opengl/src/dual_depth_peeling/doc/DualDepthPeeling.pdf
uniform mat4 World, ViewProjection;
uniform vec4 Material, Viewport;
uniform uvec4 Options;
const float EmptyDepth = 1.0, MaximumRadiance = 65504.0;
const float OverflowOpacity = 0.75;
const vec3 OverflowTint = vec3(1.0,0.0,1.0);

#ifdef EVANS_VERTEX_STAGE
// Depth capture and material replay require identical raster positions.
invariant gl_Position;
#ifdef EVANS_OIT_CAPTURE
layout(location=0) in vec3 InPosition;
layout(location=1) in vec3 InNormal;
layout(location=2) in vec4 InColor;
layout(location=3) in vec2 InUV;
out float VertexOpacity;
out vec2 UV;
void main()
{
    vec4 worldPosition = World*vec4(InPosition,1.0);
    vec4 clip = ViewProjection*worldPosition;
    clip.z = clip.z*2.0-clip.w;
    gl_Position = clip;
    VertexOpacity = InColor.a;
    UV = vec2(InUV.x,1.0-InUV.y);
}
#else
void main()
{
    vec2 corner = vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    gl_Position = vec4(corner*2.0-1.0,0.0,1.0);
}
#endif
#else
#ifdef EVANS_OIT_CAPTURE
uniform sampler2D OpaqueDepth, PreviousDepth, DiffuseMap, OpacityMap;
in float VertexOpacity;
in vec2 UV;
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float depth = gl_FragCoord.z;
    if (depth >= texelFetch(OpaqueDepth,pixel,0).r) discard;
    if (Options.x == 0u && depth <= texelFetch(PreviousDepth,pixel,0).r) discard;
    float diffuseOpacity = Options.y != 0u ? texture(DiffuseMap,UV).a : 1.0;
    float mappedOpacity = Options.z != 0u ? texture(OpacityMap,UV).r : 1.0;
    float coverage = VertexOpacity*Material.x*diffuseOpacity*mappedOpacity;
    if (isnan(coverage) || isinf(coverage)) discard;
    coverage = clamp(coverage,0.0,1.0);
    if (coverage <= 0.0 || coverage < Material.y) discard;
}
#else
uniform sampler2D LayerColor, PreviousAccumulation, SceneColor, CapturedDepth;
layout(location=0) out vec4 Result;
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 previous = texelFetch(PreviousAccumulation,pixel,0);
#ifdef EVANS_OIT_ACCUMULATE
    if (texelFetch(CapturedDepth,pixel,0).r >= EmptyDepth) { Result = previous; return; }
    vec4 layer = texelFetch(LayerColor,pixel,0);
    float remaining = clamp(previous.a,0.0,1.0);
    Result = vec4(previous.rgb+remaining*layer.rgb,remaining*(1.0-clamp(layer.a,0.0,1.0)));
#else
    vec4 background = texelFetch(SceneColor,pixel,0);
    float remaining = clamp(previous.a,0.0,1.0);
    vec3 radiance = previous.rgb+remaining*background.rgb;
    if (Options.w != 0u && texelFetch(CapturedDepth,pixel,0).r < EmptyDepth)
        radiance = mix(radiance,OverflowTint,OverflowOpacity);
    Result = vec4(clamp(radiance,0.0,MaximumRadiance),1.0-remaining+remaining*clamp(background.a,0.0,1.0));
#endif
}
#endif
#endif
