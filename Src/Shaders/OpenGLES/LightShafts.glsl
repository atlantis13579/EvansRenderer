#version 300 es
// Independent ES 3.0 radial scattering from a depth-occluded analytic light disk.
// Mitchell, "Volumetric Light Scattering as a Post-Process", GPU Gems 3 (2007).
// Texture coordinates stay bottom-left throughout. Only the CPU-projected light
// center changes from canonical top-left coordinates to the GL convention.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float ScreenTriangleExtent = 3.0;
    vec2 corner = vec2(-1.0);
    if (gl_VertexID == 1) corner.x = ScreenTriangleExtent;
    if (gl_VertexID == 2) corner.y = ScreenTriangleExtent;
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
uniform vec4 ShaftProjection; // canonical UV, depth, radius in image heights
uniform vec4 ShaftColorScale; // source RGB radiance, scattering intensity
uniform vec4 ShaftFalloffViewport; // optical decay, depth bias, width, height
uniform uvec4 ShaftConfiguration; // samples, mask/composite pass, visible source
uniform highp sampler2D ShaftOpaqueColor;
uniform highp sampler2D ShaftOpaqueDepth;
uniform highp sampler2D ShaftOcclusion;
layout(location = 0) out vec4 ShaftResult;

const uint OcclusionPass = 0u;
const float MaximumHalfRadiance = 65504.0;

float OcclusionTexel(ivec2 position)
{
    if (any(lessThan(position, ivec2(0))) || any(greaterThanEqual(position, ivec2(ShaftFalloffViewport.zw)))) return 0.0;
    return texelFetch(ShaftOcclusion, position, 0).r;
}
float OcclusionBilinear(vec2 location)
{
    // Exact texel fetches need no float-filtering extension. Four explicit
    // zero-border taps also prevent bright edge replication outside the image.
    vec2 pixel = location * ShaftFalloffViewport.zw - 0.5;
    ivec2 first = ivec2(floor(pixel));
    vec2 weight = fract(pixel);
    float lower = OcclusionTexel(first) * (1.0 - weight.x) + OcclusionTexel(first + ivec2(1, 0)) * weight.x;
    float upper = OcclusionTexel(first + ivec2(0, 1)) * (1.0 - weight.x) + OcclusionTexel(first + ivec2(1, 1)) * weight.x;
    return lower * (1.0 - weight.y) + upper * weight.y;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 uv = gl_FragCoord.xy / ShaftFalloffViewport.zw;
    vec2 sourceUV = vec2(ShaftProjection.x, 1.0 - ShaftProjection.y);
    if (ShaftConfiguration.y == OcclusionPass)
    {
        float mask = 0.0;
        // The raster stage converts canonical clip Z into GL clip Z; sampled
        // hardware depth therefore already has canonical [0,1] values.
        float depth = texelFetch(ShaftOpaqueDepth, pixel, 0).r;
        if (ShaftConfiguration.z != 0u && depth + ShaftFalloffViewport.y >= ShaftProjection.z)
        {
            vec2 delta = (uv - sourceUV) * vec2(ShaftFalloffViewport.z / ShaftFalloffViewport.w, 1.0);
            float radialProfile = max(1.0 - length(delta) / ShaftProjection.w, 0.0);
            mask = radialProfile * radialProfile;
        }
        ShaftResult = vec4(mask, 0.0, 0.0, 1.0);
        return;
    }
    vec4 surface = texelFetch(ShaftOpaqueColor, pixel, 0);
    if (ShaftConfiguration.z == 0u) { ShaftResult = surface; return; }
    float radianceIntegral = 0.0;
    for (uint sampleIndex = 0u; sampleIndex < ShaftConfiguration.x; ++sampleIndex)
    {
        float t = (float(sampleIndex) + 0.5) / float(ShaftConfiguration.x);
        vec2 sampleUV = (1.0 - t) * uv + t * sourceUV;
        radianceIntegral += OcclusionBilinear(sampleUV) * exp(-ShaftFalloffViewport.x * t);
    }
    radianceIntegral /= float(ShaftConfiguration.x);
    vec3 radiance = surface.rgb + ShaftColorScale.rgb * (ShaftColorScale.a * radianceIntegral);
    ShaftResult = vec4(min(radiance, vec3(MaximumHalfRadiance)), surface.a);
}
#endif
