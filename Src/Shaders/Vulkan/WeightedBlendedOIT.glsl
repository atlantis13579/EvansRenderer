#version 450
// McGuire and Bavoil, Weighted Blended Order-Independent Transparency (2013).
// https://jcgt.org/published/0002/02/09/
// Two native attachments store weighted radiance/opacity sums and the product
// of uncovered fractions. The pipeline blends this normalized layer over HDR.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenCorner = 3.0;
    gl_Position = vec4(gl_VertexIndex == 1 ? FullscreenCorner : -1.0,
        gl_VertexIndex == 2 ? FullscreenCorner : -1.0, 0.0, 1.0);
}
#else
layout(location = 0) out vec4 TransparentLayer;
#ifdef EVANS_WEIGHTED_SAMPLES
layout(set = 0, binding = 0) uniform sampler2DMS WeightedSum;
layout(set = 0, binding = 1) uniform sampler2DMS UncoveredFraction;
#else
layout(set = 0, binding = 0) uniform sampler2D WeightedSum;
layout(set = 0, binding = 1) uniform sampler2D UncoveredFraction;
#endif
void main()
{
    ivec2 position = ivec2(gl_FragCoord.xy);
#ifdef EVANS_WEIGHTED_SAMPLES
    // SampleRateShading and a matching multisample target retain each sample's
    // visibility. Averaging statistics first would change intersecting edges.
    vec4 sum = texelFetch(WeightedSum, position, gl_SampleID);
    float transmission = texelFetch(UncoveredFraction, position, gl_SampleID).r;
#else
    vec4 sum = texelFetch(WeightedSum, position, 0);
    float transmission = texelFetch(UncoveredFraction, position, 0).r;
#endif
    TransparentLayer = vec4(0.0);
    if (sum.a <= 0.0) return;
    const float HalfFloatSceneLimit = 65504.0;
    TransparentLayer = vec4(clamp(sum.rgb / sum.a, 0.0, HalfFloatSceneLimit),
        1.0 - clamp(transmission, 0.0, 1.0));
}
#endif
