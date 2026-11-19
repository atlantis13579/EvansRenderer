#version 450
// Vulkan centered-shutter reconstruction from independent motion vectors.
// Tile reduction and neighborhood dilation select a depth-aware gather axis.
// McGuire et al., "A Reconstruction Filter for Plausible Motion Blur" (2012):
// https://casual-effects.com/research/McGuire2012Blur/index.html
// The bounded line footprint below is a teaching variant, not the paper's
// cone/cylinder kernel. One visible depth layer cannot recover hidden surfaces.
layout(set = 0, binding = 0, std140, row_major) uniform MotionConstants
{
    mat4 World;
    mat4 ViewProjection;
    mat4 PreviousWorldViewProjection;
    mat4 InverseViewProjection;
    vec4 DepthPlane;
    vec4 ViewportExposure; // width, height, shutter / (2 * interval), radius limit
    vec4 Material; // object/material opacity, alpha cutoff
    uvec4 Options; // filter pass, diffuse texture, opacity texture, history valid
} Motion;

const float HalfPixel = 0.5;
const float MinimumHomogeneousW = 1.0e-7;
const float ClearDepth = 1.0;

#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 1) uniform texture2D SceneColor;
layout(set = 0, binding = 2) uniform texture2D SceneDepth;
layout(set = 0, binding = 3) uniform texture2D Velocity;
layout(set = 0, binding = 4) uniform texture2D TileVelocity;
layout(set = 0, binding = 5) uniform sampler PointSampler;
layout(location = 0) out vec4 OutColor;
const uint TileMaxPass = 0u;
const uint NeighborMaxPass = 1u;
const uint CompositePass = 3u;
const int TileExtent = 16;
const int TileSearchRadius = 1;
const int MaximumSamples = 33;
const float StationaryThreshold = 0.5;
const float PixelWidth = 1.0;
const float RelativeDepthTransition = 0.02;
const float MinimumDepthTransition = 1.0e-4;
const float BackgroundDistance = 1.0e20;

vec2 ShutterTravel(ivec2 pixel)
{
    vec4 motion = texelFetch(sampler2D(Velocity, PointSampler), pixel, 0);
    if (motion.w <= 0.0 || Motion.ViewportExposure.z <= 0.0) return vec2(0.0);
    vec2 travel = motion.xy * Motion.ViewportExposure.z;
    float radius = length(travel);
    return radius > Motion.ViewportExposure.w ? travel * (Motion.ViewportExposure.w / radius) : travel;
}
float EyeDistance(ivec2 pixel)
{
    float depth = texelFetch(sampler2D(SceneDepth, PointSampler), pixel, 0).r;
    if (depth >= ClearDepth) return BackgroundDistance;
    vec2 uv = (vec2(pixel) + HalfPixel) / Motion.ViewportExposure.xy;
    vec4 world = Motion.InverseViewProjection * vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);
    if (abs(world.w) <= MinimumHomogeneousW) return BackgroundDistance;
    return max(dot(Motion.DepthPlane, vec4(world.xyz / world.w, 1.0)), 0.0);
}
float InFront(float sourceDepth, float destinationDepth)
{
    float transition = max(min(sourceDepth, destinationDepth) * RelativeDepthTransition, MinimumDepthTransition);
    return clamp(1.0 - (sourceDepth - destinationDepth) / transition, 0.0, 1.0);
}
float LineFootprint(vec2 offset, vec2 travel)
{
    float radius = length(travel);
    if (radius <= StationaryThreshold) return 0.0;
    vec2 direction = travel / radius;
    float longitudinal = abs(dot(offset, direction));
    float transverse = abs(offset.x * direction.y - offset.y * direction.x);
    return clamp(radius + HalfPixel - longitudinal, 0.0, 1.0) *
        clamp(PixelWidth - transverse, 0.0, 1.0) / max(2.0 * radius, PixelWidth);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 imageExtent = ivec2(Motion.ViewportExposure.xy);
    ivec2 tileCount = (imageExtent + TileExtent - 1) / TileExtent;
    if (Motion.Options.x == TileMaxPass)
    {
        vec2 longest = vec2(0.0);
        for (int row = 0; row < TileExtent; ++row)
        for (int column = 0; column < TileExtent; ++column)
        {
            ivec2 samplePixel = pixel * TileExtent + ivec2(column, row);
            if (any(greaterThanEqual(samplePixel, imageExtent))) continue;
            vec2 candidate = ShutterTravel(samplePixel);
            if (dot(candidate, candidate) > dot(longest, longest)) longest = candidate;
        }
        OutColor = vec4(longest, 0.0, 0.0);
        return;
    }
    if (Motion.Options.x == NeighborMaxPass)
    {
        vec2 longest = vec2(0.0);
        for (int row = -TileSearchRadius; row <= TileSearchRadius; ++row)
        for (int column = -TileSearchRadius; column <= TileSearchRadius; ++column)
        {
            ivec2 tile = clamp(pixel + ivec2(column, row), ivec2(0), tileCount - 1);
            vec2 candidate = texelFetch(sampler2D(TileVelocity, PointSampler), tile, 0).xy;
            if (dot(candidate, candidate) > dot(longest, longest)) longest = candidate;
        }
        OutColor = vec4(longest, 0.0, 0.0);
        return;
    }
    vec4 center = texelFetch(sampler2D(SceneColor, PointSampler), pixel, 0);
    OutColor = center;
    if (Motion.Options.x == CompositePass) return;
    vec2 axis = texelFetch(sampler2D(TileVelocity, PointSampler), pixel / TileExtent, 0).xy;
    float radius = length(axis);
    if (radius <= StationaryThreshold) return;
    vec2 destinationTravel = ShutterTravel(pixel);
    float destinationDepth = EyeDistance(pixel);
    int halfSamples = min((MaximumSamples - 1) / 2, max(1, int(ceil(radius))));
    float spacing = radius / float(halfSamples);
    float normalization = 2.0 / max(2.0 * length(destinationTravel), PixelWidth);
    vec3 radiance = center.rgb * normalization;
    for (int tap = -halfSamples; tap <= halfSamples; ++tap)
    {
        if (tap == 0) continue;
        ivec2 source = ivec2(roundEven(vec2(pixel) + axis * (float(tap) / float(halfSamples))));
        if (any(lessThan(source, ivec2(0))) || any(greaterThanEqual(source, imageExtent)) || all(equal(source, pixel))) continue;
        vec2 offset = vec2(source - pixel);
        float sourceDepth = EyeDistance(source);
        float weight = spacing * (InFront(sourceDepth, destinationDepth) * LineFootprint(-offset, ShutterTravel(source)) +
            InFront(destinationDepth, sourceDepth) * LineFootprint(offset, destinationTravel));
        radiance += texelFetch(sampler2D(SceneColor, PointSampler), source, 0).rgb * weight;
        normalization += weight;
    }
    OutColor = vec4(radiance / normalization, center.a);
}
#endif
