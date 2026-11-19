#version 300 es
// Bavoil and Sainz, Image-Space Horizon-Based Ambient Occlusion, SIGGRAPH 2008.
// https://developer.download.nvidia.com/presentations/2008/SIGGRAPH/HBAO_SIG08b.pdf
// Quadratic falloff: https://developer.download.nvidia.com/SDK/10.5/direct3d/Source/ScreenSpaceAO/doc/ScreenSpaceAO.pdf
// March screen-space directions and integrate only newly raised sine horizons.
// A tangent correction at each snapped sample avoids false AO on tilted planes.
// Grazing/back-facing surfaces retain the single-depth heightfield approximation.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FullscreenExtent = 3.0;
    vec2 corner = gl_VertexID == 0 ? vec2(-1.0) :
        gl_VertexID == 1 ? vec2(FullscreenExtent, -1.0) : vec2(-1.0, FullscreenExtent);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
uniform mat4 ViewProjection;
uniform mat4 InverseViewProjection;
uniform vec4 ViewportRadiusBias;
uniform vec4 ForwardIntensity;
uniform uvec4 Options;
uniform sampler2D SceneDepth;
uniform sampler2D NormalRoughness;
uniform sampler2D AmbientMetallic;
uniform sampler2D PreviousAO;
out vec4 OutColor;
const uint EstimatePass = 0u;
const uint HorizontalPass = 1u;
const uint VerticalPass = 2u;
const uint CompositePass = 3u;
const int HorizonDirectionCount = 8;
const int HorizonStepCount = 8;
const int BilateralRadius = 2;
const float Pi = 3.141592653589793;
const float NormalizeEpsilon = 1.0e-12;
const float ProjectionEpsilon = 1.0e-7;
const float TangentDenominatorEpsilon = 1.0e-6;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float NormalSimilarityExponent = 8.0;
const float MinimumFilterWeight = 1.0e-7;
const float PixelNoiseScale = 52.9829189;
const vec2 PixelNoiseDirection = vec2(0.06711056, 0.00583715);

bool WorldPosition(vec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (pixel + 0.5) * ViewportRadiusBias.xy;
    // GL window depth already matches the original camera's [0,1] depth range.
    vec4 world = InverseViewProjection * vec4(uv * 2.0 - 1.0, depth, 1.0);
    position = vec3(0.0);
    if (abs(world.w) < ProjectionEpsilon) return false;
    position = world.xyz / world.w;
    return !any(isnan(position)) && !any(isinf(position));
}

bool Surface(ivec2 pixel, out vec3 position, out vec3 normal)
{
    float depth = texelFetch(SceneDepth, pixel, 0).r;
    vec4 stored = texelFetch(NormalRoughness, pixel, 0);
    float squared = dot(stored.xyz, stored.xyz);
    position = vec3(0.0);
    normal = vec3(0.0, 1.0, 0.0);
    if (depth >= 1.0 || stored.w < 0.0 || squared <= NormalizeEpsilon) return false;
    normal = stored.xyz * inversesqrt(squared);
    return WorldPosition(vec2(pixel), depth, position);
}

float HorizonVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    vec2 canonicalPixel = vec2(pixel.x, extent.y - 1 - pixel.y);
    float rotation = 2.0 * Pi * fract(PixelNoiseScale * fract(dot(canonicalPixel, PixelNoiseDirection)));
    vec3 towardViewer = -ForwardIntensity.xyz;
    float centerDepth = texelFetch(SceneDepth, pixel, 0).r;
    float radius = ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float obstruction = 0.0;
    for (int directionIndex = 0; directionIndex < HorizonDirectionCount; ++directionIndex)
    {
        float angle = rotation + 2.0 * Pi * float(directionIndex) / float(HorizonDirectionCount);
        // Canonical directions are top-left; GL texels grow upward instead.
        vec2 screenDirection = vec2(cos(angle), -sin(angle));
        vec3 adjacent;
        if (!WorldPosition(vec2(pixel) + screenDirection, centerDepth, adjacent)) continue;
        vec3 planeDirection = adjacent - position;
        planeDirection -= towardViewer * dot(planeDirection, towardViewer);
        float planeSquared = dot(planeDirection, planeDirection);
        if (planeSquared <= NormalizeEpsilon) continue;
        planeDirection *= inversesqrt(planeSquared);
        vec4 endpoint = ViewProjection * vec4(position + radius * planeDirection, 1.0);
        if (any(isnan(endpoint)) || any(isinf(endpoint)) || endpoint.w <= ProjectionEpsilon) continue;
        vec2 endpointUV = endpoint.xy / endpoint.w * 0.5 + 0.5;
        vec2 pixelOffset = endpointUV * vec2(extent) - (vec2(pixel) + 0.5);
        if (any(isnan(pixelOffset)) || any(isinf(pixelOffset))) continue;
        float highestCorrectedHorizon = 0.0;
        ivec2 previousPixel = pixel;
        for (int stepIndex = 1; stepIndex <= HorizonStepCount; ++stepIndex)
        {
            float fraction = float(stepIndex) / float(HorizonStepCount);
            vec2 snapped = floor(vec2(pixel) + pixelOffset * fraction + 0.5);
            // Bound floating coordinates before integer conversion at the frustum edge.
            if (any(lessThan(snapped, vec2(0.0))) || any(greaterThanEqual(snapped, vec2(extent)))) continue;
            ivec2 samplePixel = ivec2(snapped);
            if (all(equal(samplePixel, previousPixel))) continue;
            previousPixel = samplePixel;
            if (all(equal(samplePixel, pixel))) continue;
            float sampleDepth = texelFetch(SceneDepth, samplePixel, 0).r;
            if (sampleDepth >= 1.0) continue;
            vec3 samplePosition;
            if (!WorldPosition(vec2(samplePixel), sampleDepth, samplePosition)) continue;
            vec3 delta = samplePosition - position;
            float distanceSquared = dot(delta, delta);
            if (distanceSquared <= NormalizeEpsilon || distanceSquared >= radiusSquared) continue;
            float height = dot(delta, towardViewer);
            vec3 snappedDirection = delta - towardViewer * height;
            float snappedSquared = dot(snappedDirection, snappedDirection);
            if (snappedSquared <= NormalizeEpsilon) continue;
            snappedDirection *= inversesqrt(snappedSquared);
            float tangentAngle = atan(-dot(normal, snappedDirection),
                max(dot(normal, towardViewer), TangentDenominatorEpsilon));
            float biasedTangent = clamp(tangentAngle + ViewportRadiusBias.w, -Pi * 0.5, Pi * 0.5);
            float sineHorizon = clamp(height * inversesqrt(distanceSquared), -1.0, 1.0);
            float correctedHorizon = max(sineHorizon - sin(biasedTangent), 0.0);
            float increment = max(correctedHorizon - highestCorrectedHorizon, 0.0);
            obstruction += increment * (1.0 - distanceSquared / radiusSquared);
            highestCorrectedHorizon = max(highestCorrectedHorizon, correctedHorizon);
        }
    }
    // Rejected directions stay open instead of changing the normalization.
    return clamp(1.0 - ForwardIntensity.w * obstruction / float(HorizonDirectionCount), 0.0, 1.0);
}

float FilterVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    ivec2 direction = Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float total = 0.0;
    float totalWeight = 0.0;
    for (int offset = -BilateralRadius; offset <= BilateralRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + direction * offset, ivec2(0), extent - 1);
        vec3 neighborPosition, neighborNormal;
        if (!Surface(neighbor, neighborPosition, neighborNormal)) continue;
        float depthDelta = dot(neighborPosition - position, ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialSigma * SpatialSigma));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0 * depthSigma * depthSigma));
        float normalWeight = pow(clamp(dot(neighborNormal, normal), 0.0, 1.0), NormalSimilarityExponent);
        float weight = spatialWeight * depthWeight * normalWeight;
        total += texelFetch(PreviousAO, neighbor, 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > MinimumFilterWeight ? clamp(total / totalWeight, 0.0, 1.0) : 1.0;
}

void main()
{
    // Avoid approximate filter division disturbing the exact identity setting.
    if (ForwardIntensity.w <= 0.0)
    {
        OutColor = Options.x == CompositePass ? vec4(0.0) : vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 position, normal;
    bool valid = Surface(pixel, position, normal);
    if (Options.x == EstimatePass)
        OutColor = vec4(valid ? HorizonVisibility(pixel, position, normal) : 1.0, 0.0, 0.0, 1.0);
    else if (Options.x == HorizontalPass || Options.x == VerticalPass)
        OutColor = vec4(valid ? FilterVisibility(pixel, position, normal) : 1.0, 0.0, 0.0, 1.0);
    else
    {
        if (!valid) { OutColor = vec4(0.0); return; }
        float visibility = clamp(texelFetch(PreviousAO, pixel, 0).r, 0.0, 1.0);
        vec3 ambient = max(texelFetch(AmbientMetallic, pixel, 0).rgb, vec3(0.0));
        // RGB reverse-subtract blending preserves direct light, emission and alpha.
        OutColor = vec4(ambient * (1.0 - visibility), 0.0);
    }
}
#endif
