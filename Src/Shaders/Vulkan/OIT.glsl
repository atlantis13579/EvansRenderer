#version 450
// Vulkan GLSL stays readable source and compiles in memory at runtime. Capture
// uses native [0,1] depth and reverses clip Y to match this renderer's raster ABI.
// One front layer per iteration, with a final capture-only overflow probe.
// Everitt (2001), Interactive Order-Independent Transparency; Bavoil/Myers (2008):
// https://developer.download.nvidia.com/SDK/10/opengl/src/dual_depth_peeling/doc/DualDepthPeeling.pdf
layout(set=0,binding=0,std140,row_major) uniform PeelingUniforms
{
    mat4 World, ViewProjection;
    vec4 Material, Viewport;
    uvec4 Options;
};
const float ClearDepth = 1.0, HalfFloatLimit = 65504.0;
const vec3 OverflowColor = vec3(1.0,0.0,1.0);
const float DiagnosticCoverage = 0.75;

#if defined(EVANS_VERTEX_STAGE)
// Depth capture and material replay require identical raster positions.
invariant gl_Position;
precise gl_Position;
#if defined(EVANS_OIT_CAPTURE)
layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 VertexColor;
layout(location=3) in vec2 Coordinates;
layout(location=0) out float Opacity;
layout(location=1) out vec2 UV;
void main()
{
    vec4 world = World*vec4(Position,1.0);
    gl_Position = ViewProjection*world;
    gl_Position.y = -gl_Position.y;
    Opacity = VertexColor.a;
    UV = vec2(Coordinates.x,1.0-Coordinates.y);
}
#else
void main()
{
    vec2 triangle = vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position = vec4(triangle*2.0-1.0,0.0,1.0);
}
#endif
#else
layout(set=0,binding=1) uniform sampler MaterialSampler;
#if defined(EVANS_OIT_CAPTURE)
layout(set=0,binding=2) uniform texture2D OpaqueDepth;
layout(set=0,binding=3) uniform texture2D PreviousDepth;
layout(set=0,binding=4) uniform texture2D DiffuseMap;
layout(set=0,binding=5) uniform texture2D OpacityMap;
layout(location=0) in float Opacity;
layout(location=1) in vec2 UV;
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float depth = gl_FragCoord.z;
    if (depth >= texelFetch(sampler2D(OpaqueDepth,MaterialSampler),pixel,0).r) discard;
    if (Options.x == 0u && depth <= texelFetch(sampler2D(PreviousDepth,MaterialSampler),pixel,0).r) discard;
    float coverage = Opacity*Material.x;
    if (Options.y != 0u) coverage *= texture(sampler2D(DiffuseMap,MaterialSampler),UV).a;
    if (Options.z != 0u) coverage *= texture(sampler2D(OpacityMap,MaterialSampler),UV).r;
    if (isnan(coverage) || isinf(coverage)) discard;
    coverage = clamp(coverage,0.0,1.0);
    if (coverage <= 0.0 || coverage < Material.y) discard;
}
#else
layout(set=0,binding=6) uniform texture2D LayerColor;
layout(set=0,binding=7) uniform texture2D PreviousAccumulation;
layout(set=0,binding=8) uniform texture2D SceneColor;
layout(set=0,binding=9) uniform texture2D CapturedDepth;
layout(location=0) out vec4 Result;
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 previous = texelFetch(sampler2D(PreviousAccumulation,MaterialSampler),pixel,0);
#if defined(EVANS_OIT_ACCUMULATE)
    if (texelFetch(sampler2D(CapturedDepth,MaterialSampler),pixel,0).r >= ClearDepth) { Result = previous; return; }
    vec4 layer = texelFetch(sampler2D(LayerColor,MaterialSampler),pixel,0);
    float transmittance = clamp(previous.a,0.0,1.0);
    Result = vec4(previous.rgb+transmittance*layer.rgb,transmittance*(1.0-clamp(layer.a,0.0,1.0)));
#else
    vec4 background = texelFetch(sampler2D(SceneColor,MaterialSampler),pixel,0);
    float transmittance = clamp(previous.a,0.0,1.0);
    vec3 radiance = previous.rgb+transmittance*background.rgb;
    if (Options.w != 0u && texelFetch(sampler2D(CapturedDepth,MaterialSampler),pixel,0).r < ClearDepth)
        radiance = mix(radiance,OverflowColor,DiagnosticCoverage);
    Result = vec4(clamp(radiance,0.0,HalfFloatLimit),1.0-transmittance+transmittance*clamp(background.a,0.0,1.0));
#endif
}
#endif
#endif
