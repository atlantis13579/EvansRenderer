#version 300 es
// McGuire and Bavoil, Weighted Blended Order-Independent Transparency (2013).
// https://jcgt.org/published/0002/02/09/
// This ES program normalizes weighted radiance and resolves product revealage.
// Fixed-function blending places the resulting straight color over opaque HDR.
precision highp float;
precision highp int;

#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float Overscan = 3.0;
    vec2 corner = vec2(gl_VertexID == 1 ? Overscan : -1.0,
        gl_VertexID == 2 ? Overscan : -1.0);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
layout(location = 0) out vec4 ResolvedTransparency;
#ifdef EVANS_WEIGHTED_SAMPLES
uniform highp sampler2DMS AccumulatedRadiance;
uniform highp sampler2DMS Revealage;
uniform int ResolveSample;
#else
uniform highp sampler2D AccumulatedRadiance;
uniform highp sampler2D Revealage;
#endif
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#ifdef EVANS_WEIGHTED_SAMPLES
    // The host restricts each draw to this sample's coverage bit. Normalize
    // before the final color average to retain correlated silhouette coverage.
    vec4 accumulated = texelFetch(AccumulatedRadiance, pixel, ResolveSample);
    float visibility = texelFetch(Revealage, pixel, ResolveSample).r;
#else
    vec4 accumulated = texelFetch(AccumulatedRadiance, pixel, 0);
    float visibility = texelFetch(Revealage, pixel, 0).r;
#endif
    ResolvedTransparency = vec4(0.0);
    if (accumulated.a <= 0.0) return;
    const float MaximumSceneRadiance = 65504.0; // RGBA16F scene target limit.
    vec3 radiance = clamp(accumulated.rgb / accumulated.a, 0.0, MaximumSceneRadiance);
    ResolvedTransparency = vec4(radiance, 1.0 - clamp(visibility, 0.0, 1.0));
}
#endif
