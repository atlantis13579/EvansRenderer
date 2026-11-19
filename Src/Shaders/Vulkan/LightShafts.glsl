#version 450
// Native Vulkan light-shaft reconstruction, independently authored for this API.
// Kenny Mitchell, "Volumetric Light Scattering as a Post-Process", GPU Gems 3.
// Mask and HDR targets use top-left pixel coordinates with positive viewports;
// the source projection uses the same convention as the surface clip-Y flip.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float OversizedTriangleEdge = 3.0;
    vec2 vertices[3] = vec2[3](vec2(-1.0, -1.0), vec2(OversizedTriangleEdge, -1.0), vec2(-1.0, OversizedTriangleEdge));
    gl_Position = vec4(vertices[gl_VertexIndex], 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform ShaftFrame
{
    vec4 ProjectedDisk; // source UV, canonical depth, radius in image heights
    vec4 LightAndGain; // source RGB radiance, scattering strength
    vec4 DecayBiasExtent; // optical decay, depth comparison bias, viewport XY
    uvec4 SamplesStageVisible;
} Shaft;
layout(set = 0, binding = 1) uniform sampler2D InputColor;
layout(set = 0, binding = 2) uniform sampler2D InputDepth;
layout(set = 0, binding = 3) uniform sampler2D InputMask;
layout(location = 0) out vec4 OutputColor;

const uint MakeOcclusionMask = 0u;
const float HalfStorageCeiling = 65504.0;

float MaskSample(ivec2 location)
{
    ivec2 size = textureSize(InputMask, 0);
    if (any(lessThan(location, ivec2(0))) || any(greaterThanEqual(location, size))) return 0.0;
    return texelFetch(InputMask, location, 0).r;
}
float FilterMask(vec2 uv)
{
    // Four exact texel reads implement a zero-border bilinear filter without
    // sampler interpolation precision or replicated edge radiance.
    vec2 texelPosition = uv * Shaft.DecayBiasExtent.zw - 0.5;
    ivec2 lower = ivec2(floor(texelPosition));
    vec2 amount = fract(texelPosition);
    vec2 rows = vec2(
        mix(MaskSample(lower), MaskSample(lower + ivec2(1, 0)), amount.x),
        mix(MaskSample(lower + ivec2(0, 1)), MaskSample(lower + ivec2(1, 1)), amount.x));
    return mix(rows.x, rows.y, amount.y);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 uv = gl_FragCoord.xy / Shaft.DecayBiasExtent.zw;
    if (Shaft.SamplesStageVisible.y == MakeOcclusionMask)
    {
        float visibleDisk = 0.0;
        float surfaceDepth = texelFetch(InputDepth, pixel, 0).r;
        if (Shaft.SamplesStageVisible.z != 0u && surfaceDepth + Shaft.DecayBiasExtent.y >= Shaft.ProjectedDisk.z)
        {
            vec2 offset = uv - Shaft.ProjectedDisk.xy;
            offset.x *= Shaft.DecayBiasExtent.z / Shaft.DecayBiasExtent.w;
            float profile = clamp(1.0 - length(offset) / Shaft.ProjectedDisk.w, 0.0, 1.0);
            visibleDisk = profile * profile;
        }
        OutputColor = vec4(visibleDisk, 0.0, 0.0, 1.0);
        return;
    }
    vec4 opaque = texelFetch(InputColor, pixel, 0);
    if (Shaft.SamplesStageVisible.z == 0u) { OutputColor = opaque; return; }
    float transport = 0.0;
    uint samples = Shaft.SamplesStageVisible.x;
    for (uint index = 0u; index < samples; ++index)
    {
        float t = (float(index) + 0.5) / float(samples);
        transport += FilterMask(mix(uv, Shaft.ProjectedDisk.xy, t)) * exp(-Shaft.DecayBiasExtent.x * t);
    }
    // The quadrature integrates over [0,1]. More samples refine silhouettes,
    // rather than multiplying the source exposure. Alpha stays unchanged.
    transport *= Shaft.LightAndGain.a / float(samples);
    OutputColor = vec4(min(opaque.rgb + Shaft.LightAndGain.rgb * transport, vec3(HalfStorageCeiling)), opaque.a);
}
#endif
