#version 300 es
// ES centered-shutter reconstruction from independent motion vectors.
// McGuire et al., "A Reconstruction Filter for Plausible Motion Blur" (2012):
// https://casual-effects.com/research/McGuire2012Blur/index.html
// Tile dilation chooses a sampling direction; original source/destination motion
// and soft depth classification determine contribution. Our line-footprint box
// shutter differs from the paper's cone/cylinder reconstruction kernels.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform vec4 ExtentExposure; // full width/height, shutter/(2*interval), maximum radius
uniform uvec4 Options; // fullscreen stage, diffuse map, opacity map, valid history
const float PixelCenter = 0.5;
const float MinimumClipW = 1.0e-7;
const float FarDepth = 1.0;

#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenTriangleExtent = 3.0;
    vec2 position = gl_VertexID == 0 ? vec2(-1.0) : gl_VertexID == 1 ?
        vec2(FullscreenTriangleExtent, -1.0) : vec2(-1.0, FullscreenTriangleExtent);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
uniform mat4 InverseCamera;
uniform vec4 OpticalDepthPlane;
uniform sampler2D OpaqueRadiance;
uniform sampler2D OpaqueDepth;
uniform sampler2D VelocityMap;
uniform sampler2D MaximumMap;
layout(location = 0) out vec4 ResultColor;
const uint TileReduction = 0u;
const uint NeighborReduction = 1u;
const uint CopyReconstruction = 3u;
const int TileSize = 16;
const int NeighborTileRadius = 1;
const int MaximumGatherSamples = 33;
const float StationaryRadius = 0.5;
const float PixelWidth = 1.0;
const float RelativeDepthSoftness = 0.02;
const float AbsoluteDepthSoftness = 1.0e-4;
const float BackgroundDistance = 1.0e20;

ivec2 NativePixel(ivec2 logicalPixel, ivec2 extent)
{
    return ivec2(logicalPixel.x, extent.y - 1 - logicalPixel.y);
}
vec2 ExposureTravel(ivec2 pixel)
{
    vec4 velocity = texelFetch(VelocityMap, NativePixel(pixel, ivec2(ExtentExposure.xy)), 0);
    if (velocity.w <= 0.0 || ExtentExposure.z <= 0.0) return vec2(0.0);
    vec2 travel = velocity.xy * ExtentExposure.z;
    float radius = length(travel);
    return radius > ExtentExposure.w ? travel * (ExtentExposure.w / radius) : travel;
}
float OpticalDistance(ivec2 pixel)
{
    float depth = texelFetch(OpaqueDepth, NativePixel(pixel, ivec2(ExtentExposure.xy)), 0).r;
    if (depth >= FarDepth) return BackgroundDistance;
    vec2 uv = (vec2(pixel) + PixelCenter) / ExtentExposure.xy;
    vec4 world = InverseCamera * vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);
    if (abs(world.w) <= MinimumClipW) return BackgroundDistance;
    return max(dot(OpticalDepthPlane, vec4(world.xyz / world.w, 1.0)), 0.0);
}
float ForegroundWeight(float sourceDepth, float destinationDepth)
{
    float softness = max(min(sourceDepth, destinationDepth) * RelativeDepthSoftness, AbsoluteDepthSoftness);
    return clamp(1.0 - (sourceDepth - destinationDepth) / softness, 0.0, 1.0);
}
float SweptPixelWeight(vec2 offset, vec2 travel)
{
    float radius = length(travel);
    if (radius <= StationaryRadius) return 0.0;
    vec2 direction = travel / radius;
    float along = abs(dot(offset, direction));
    float across = abs(offset.x * direction.y - offset.y * direction.x);
    float coverage = clamp(radius + PixelCenter - along, 0.0, 1.0) * clamp(PixelWidth - across, 0.0, 1.0);
    return coverage / max(2.0 * radius, PixelWidth);
}
void main()
{
    ivec2 fullSize = ivec2(ExtentExposure.xy);
    ivec2 tileCount = (fullSize + TileSize - 1) / TileSize;
    bool reducing = Options.x == TileReduction || Options.x == NeighborReduction;
    ivec2 pixel = NativePixel(ivec2(gl_FragCoord.xy), reducing ? tileCount : fullSize);
    if (Options.x == TileReduction)
    {
        vec2 maximum = vec2(0.0);
        for (int y = 0; y < TileSize; ++y)
        for (int x = 0; x < TileSize; ++x)
        {
            ivec2 source = pixel * TileSize + ivec2(x, y);
            if (any(greaterThanEqual(source, fullSize))) continue;
            vec2 candidate = ExposureTravel(source);
            if (dot(candidate, candidate) > dot(maximum, maximum)) maximum = candidate;
        }
        ResultColor = vec4(maximum, 0.0, 0.0);
        return;
    }
    if (Options.x == NeighborReduction)
    {
        vec2 maximum = vec2(0.0);
        for (int y = -NeighborTileRadius; y <= NeighborTileRadius; ++y)
        for (int x = -NeighborTileRadius; x <= NeighborTileRadius; ++x)
        {
            ivec2 tile = clamp(pixel + ivec2(x, y), ivec2(0), tileCount - 1);
            vec2 candidate = texelFetch(MaximumMap, NativePixel(tile, tileCount), 0).xy;
            if (dot(candidate, candidate) > dot(maximum, maximum)) maximum = candidate;
        }
        ResultColor = vec4(maximum, 0.0, 0.0);
        return;
    }
    vec4 center = texelFetch(OpaqueRadiance, NativePixel(pixel, fullSize), 0);
    ResultColor = center;
    if (Options.x == CopyReconstruction) return;
    vec2 dominant = texelFetch(MaximumMap, NativePixel(pixel / TileSize, tileCount), 0).xy;
    float radius = length(dominant);
    if (radius <= StationaryRadius) return;
    vec2 ownTravel = ExposureTravel(pixel);
    float ownDepth = OpticalDistance(pixel);
    int halfSamples = min((MaximumGatherSamples - 1) / 2, max(1, int(ceil(radius))));
    float spacing = radius / float(halfSamples);
    float totalWeight = 2.0 / max(2.0 * length(ownTravel), PixelWidth);
    vec3 accumulated = center.rgb * totalWeight;
    for (int sampleIndex = -halfSamples; sampleIndex <= halfSamples; ++sampleIndex)
    {
        if (sampleIndex == 0) continue;
        ivec2 source = ivec2(roundEven(vec2(pixel) + dominant * (float(sampleIndex) / float(halfSamples))));
        if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, fullSize)) || all(equal(source, pixel))) continue;
        vec2 offset = vec2(source - pixel);
        float sourceDepth = OpticalDistance(source);
        float weight = spacing * (ForegroundWeight(sourceDepth, ownDepth) * SweptPixelWeight(-offset, ExposureTravel(source)) +
            ForegroundWeight(ownDepth, sourceDepth) * SweptPixelWeight(offset, ownTravel));
        accumulated += texelFetch(OpaqueRadiance, NativePixel(source, fullSize), 0).rgb * weight;
        totalWeight += weight;
    }
    ResultColor = vec4(accumulated / totalWeight, center.a);
}
#endif
