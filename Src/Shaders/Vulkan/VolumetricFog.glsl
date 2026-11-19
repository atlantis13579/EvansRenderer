#version 450
// Vulkan height-medium integration, independent of the other API sources.
// Prefix optical depth solves the constant-source transport equation:
// T = exp(-integral(sigmaT ds)), S = sourceRadiance * (1 - T).
// Pharr, Jakob, Humphreys, Physically Based Rendering, "Transmittance".
// https://pbr-book.org/4ed/Volume_Scattering/Transmittance
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenTriangleExtent = 3.0;
    vec2 corner = gl_VertexIndex == 0 ? vec2(-1.0) : gl_VertexIndex == 1 ?
        vec2(FullscreenTriangleExtent, -1.0) : vec2(-1.0, FullscreenTriangleExtent);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform FogParameters
{
    mat4 InverseCamera;
    vec4 DensityHeightRange;
    vec4 AmbientSource;
    vec4 ViewAndGrid;
    uvec4 StageSlicesSamplesEnabled;
} Fog;
layout(set = 0, binding = 1) uniform sampler2D SceneRadiance;
layout(set = 0, binding = 2) uniform sampler2D SceneDepth;
layout(set = 0, binding = 3) uniform sampler2D CumulativeMedium;
layout(location = 0) out vec4 Result;

const uint IntegrateStage = 0u;
const float SampleMidpoint = 0.5;
const float MinimumRayLength = 1.0e-6;
const float SaturatedOpticalDepth = 80.0;
const float MaximumStoredRadiance = 65504.0;
const vec4 IdentityMedium = vec4(0.0, 0.0, 0.0, 1.0);

vec3 PositionOnRay(vec2 uv, float depth)
{
    // The surface vertex shaders flip clip Y. Positive-height Vulkan viewports
    // give top-left image coordinates, so undo that flip before unprojection.
    vec2 ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    vec4 point = Fog.InverseCamera * vec4(ndc, depth, 1.0);
    return point.xyz / point.w;
}
vec4 IntegratePrefix(ivec2 texel)
{
    int cellsPerSlice = int(Fog.ViewAndGrid.z);
    uint boundary = uint(texel.x / cellsPerSlice) + 1u;
    vec2 uv = (vec2(texel.x % cellsPerSlice, texel.y) + SampleMidpoint) / Fog.ViewAndGrid.zw;
    vec3 origin = PositionOnRay(uv, 0.0);
    vec3 ray = PositionOnRay(uv, 1.0) - origin;
    float rayLength = max(length(ray), MinimumRayLength);
    float segmentLength = min(rayLength, Fog.DensityHeightRange.w) /
        float(Fog.StageSlicesSamplesEnabled.y * Fog.StageSlicesSamplesEnabled.z);
    float heightStep = ray.y * segmentLength / rayLength;
    float opticalDepth = 0.0;
    uint sampleCount = boundary * Fog.StageSlicesSamplesEnabled.z;
    for (uint sampleIndex = 0u; sampleIndex < sampleCount; ++sampleIndex)
    {
        float height = origin.y + (float(sampleIndex) + SampleMidpoint) * heightStep;
        float exponent = Fog.DensityHeightRange.z * max(height - Fog.DensityHeightRange.y, 0.0);
        float localExtinction = Fog.DensityHeightRange.x * exp(-min(exponent, SaturatedOpticalDepth));
        opticalDepth += localExtinction * segmentLength;
    }
    float transmission = exp(-min(opticalDepth, SaturatedOpticalDepth));
    return vec4(Fog.AmbientSource.rgb * (1.0 - transmission), transmission);
}
vec4 ReadPrefix(vec2 uv, uint boundary)
{
    if (boundary == 0u) return IdentityMedium;
    vec2 border = SampleMidpoint / Fog.ViewAndGrid.zw;
    vec2 localUV = clamp(uv, border, 1.0 - border);
    localUV.x = (float(boundary - 1u) + localUV.x) / float(Fog.StageSlicesSamplesEnabled.y);
    return textureLod(CumulativeMedium, localUV, 0.0);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Fog.StageSlicesSamplesEnabled.x == IntegrateStage)
    {
        Result = IntegratePrefix(pixel);
        return;
    }
    vec2 uv = gl_FragCoord.xy / Fog.ViewAndGrid.xy;
    vec3 origin = PositionOnRay(uv, 0.0);
    float range = min(length(PositionOnRay(uv, 1.0) - origin), Fog.DensityHeightRange.w);
    float distance = length(PositionOnRay(uv, texelFetch(SceneDepth, pixel, 0).r) - origin);
    float boundary = clamp(distance / max(range, MinimumRayLength), 0.0, 1.0) * float(Fog.StageSlicesSamplesEnabled.y);
    uint lower = uint(boundary), upper = min(lower + 1u, Fog.StageSlicesSamplesEnabled.y);
    vec4 medium = mix(ReadPrefix(uv, lower), ReadPrefix(uv, upper), fract(boundary));
    vec4 scene = texelFetch(SceneRadiance, pixel, 0);
    Result = vec4(clamp(scene.rgb * medium.a + medium.rgb, 0.0, MaximumStoredRadiance), scene.a);
}
#endif
