#version 300 es
// Native ES 3.0 cumulative height-medium transport. Texels and gl_FragCoord
// use the GL bottom-left origin. Geometry converts canonical clip Z to GL clip
// Z, so sampled hardware depth is already the canonical [0,1] value.
// Beer-Lambert transmission and a constant isotropic ambient source:
// Pharr, Jakob, Humphreys, Physically Based Rendering, "Transmittance".
// https://pbr-book.org/4ed/Volume_Scattering/Transmittance
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenReach = 3.0;
    vec2 location = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
        gl_VertexID == 1 ? vec2(FullscreenReach, -1.0) : vec2(-1.0, FullscreenReach);
    gl_Position = vec4(location, 0.0, 1.0);
}
#else
uniform mat4 FogInverseCamera;
uniform vec4 FogMedium; // extinction, base height, height falloff, distance limit
uniform vec4 FogRadiance;
uniform vec4 FogViewGrid; // viewport XY and logical cell grid XY
uniform uvec4 FogControl; // integration/composition, depth slices, samples, enabled
uniform highp sampler2D FogSceneColor;
uniform highp sampler2D FogSceneDepth;
uniform highp sampler2D FogTexture;
layout(location = 0) out vec4 OutTransport;

const uint IntegrationStage = 0u;
const float MidpointOffset = 0.5;
const float MinimumSegmentExtent = 1.0e-6;
const float OpaqueOpticalDepth = 80.0;
const float HalfFloatCeiling = 65504.0;
const vec4 VacuumTransport = vec4(0.0, 0.0, 0.0, 1.0);

vec3 ReconstructMediumPoint(vec2 imageUV, float hardwareDepth)
{
    vec4 position = FogInverseCamera * vec4(imageUV * 2.0 - 1.0, hardwareDepth, 1.0);
    return position.xyz / position.w;
}
vec4 IntegrateCell(ivec2 atlasPixel)
{
    int gridWidth = int(FogViewGrid.z);
    uint terminalSlice = uint(atlasPixel.x / gridWidth) + 1u;
    vec2 imageUV = (vec2(atlasPixel.x % gridWidth, atlasPixel.y) + MidpointOffset) / FogViewGrid.zw;
    vec3 nearPosition = ReconstructMediumPoint(imageUV, 0.0);
    vec3 directionToFar = ReconstructMediumPoint(imageUV, 1.0) - nearPosition;
    float rayLength = max(length(directionToFar), MinimumSegmentExtent);
    float intervalDistance = min(rayLength, FogMedium.w) / float(FogControl.y * FogControl.z);
    vec3 stepVector = directionToFar * (intervalDistance / rayLength);
    vec4 accumulated = VacuumTransport;
    uint intervals = terminalSlice * FogControl.z;
    for (uint interval = 0u; interval < intervals; ++interval)
    {
        float midpointHeight = nearPosition.y + (float(interval) + MidpointOffset) * stepVector.y;
        float heightExponent = FogMedium.z * max(midpointHeight - FogMedium.y, 0.0);
        float extinction = FogMedium.x * exp(-min(heightExponent, OpaqueOpticalDepth));
        float segmentTransmission = exp(-min(extinction * intervalDistance, OpaqueOpticalDepth));
        float removedTransmission = accumulated.a * (1.0 - segmentTransmission);
        accumulated.rgb += removedTransmission * FogRadiance.rgb;
        accumulated.a *= segmentTransmission;
    }
    return accumulated;
}
vec4 SampleBoundary(vec2 imageUV, uint endpoint)
{
    if (endpoint == 0u) return VacuumTransport;
    vec2 inset = MidpointOffset / FogViewGrid.zw;
    vec2 boundedUV = clamp(imageUV, inset, 1.0 - inset);
    vec2 atlasUV = vec2((float(endpoint - 1u) + boundedUV.x) / float(FogControl.y), boundedUV.y);
    return textureLod(FogTexture, atlasUV, 0.0);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (FogControl.x == IntegrationStage)
    {
        OutTransport = IntegrateCell(pixel);
        return;
    }
    vec2 imageUV = gl_FragCoord.xy / FogViewGrid.xy;
    vec3 nearPosition = ReconstructMediumPoint(imageUV, 0.0);
    float extent = min(length(ReconstructMediumPoint(imageUV, 1.0) - nearPosition), FogMedium.w);
    float depth = texelFetch(FogSceneDepth, pixel, 0).r;
    float distanceToSurface = length(ReconstructMediumPoint(imageUV, depth) - nearPosition);
    float endpoint = clamp(distanceToSurface / max(extent, MinimumSegmentExtent), 0.0, 1.0) * float(FogControl.y);
    uint before = uint(endpoint);
    uint after = min(before + 1u, FogControl.y);
    vec4 medium = mix(SampleBoundary(imageUV, before), SampleBoundary(imageUV, after), endpoint - float(before));
    vec4 surface = texelFetch(FogSceneColor, pixel, 0);
    OutTransport = vec4(clamp(surface.rgb * medium.a + medium.rgb, 0.0, HalfFloatCeiling), surface.a);
}
#endif
