#version 450
// Independent Vulkan HBAO: outward depth searches track increasing horizons
// above the surface tangent, followed by bilateral filtering and ambient-only
// composition. No random texture or scene asset is required.
// Bavoil, Sainz and Dimitrov, "Image-Space Horizon-Based Ambient Occlusion"
// (SIGGRAPH 2008): https://rdimitrov.twistedsanity.net/HBAO_SIGGRAPH08.pdf
// Quadratic distance attenuation follows NVIDIA's Screen Space AO sample:
// https://developer.download.nvidia.com/SDK/10.5/direct3d/Source/ScreenSpaceAO/doc/ScreenSpaceAO.pdf
// The tangent follows each snapped sample's azimuth to avoid tilted-plane
// self-occlusion. Grazing surfaces retain the original heightfield approximation.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform HorizonConstants
{
    mat4 ViewProjection;
    mat4 InverseViewProjection;
    vec4 ViewportRadiusBias;
    vec4 ForwardIntensity;
    uvec4 Options;
} Horizon;
layout(set = 0, binding = 1) uniform texture2D OpaqueDepth;
layout(set = 0, binding = 2) uniform texture2D NormalRoughness;
layout(set = 0, binding = 3) uniform texture2D AmbientMetallic;
layout(set = 0, binding = 4) uniform texture2D PreviousVisibility;
layout(set = 0, binding = 5) uniform sampler PointSampler;
layout(location = 0) out vec4 OutValue;

const uint EstimatePass = 0u;
const uint HorizontalPass = 1u;
const uint CompositePass = 3u;
const int DirectionCount = 8;
const int StepsPerDirection = 8;
const int FilterRadius = 2;
const float Pi = 3.141592653589793;
const float HalfPi = Pi * 0.5;
const float PixelCenter = 0.5;
const float ClearDepth = 1.0;
const float HomogeneousEpsilon = 1.0e-7;
const float SquaredLengthEpsilon = 1.0e-12;
const float TangentNormalEpsilon = 1.0e-6;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float NormalWeightExponent = 8.0;
const float FilterWeightEpsilon = 1.0e-7;
const vec2 RotationHashDirection = vec2(0.06711056, 0.00583715);
const float RotationHashScale = 52.9829189;

float DepthAt(ivec2 pixel)
{
    return texelFetch(sampler2D(OpaqueDepth, PointSampler), pixel, 0).r;
}
vec4 NormalAt(ivec2 pixel)
{
    return texelFetch(sampler2D(NormalRoughness, PointSampler), pixel, 0);
}
bool IsLitSurface(float depth, vec4 materialNormal)
{
    return depth < ClearDepth && materialNormal.w >= 0.0 &&
        dot(materialNormal.xyz, materialNormal.xyz) > SquaredLengthEpsilon;
}
vec3 UnitVector(vec3 vector)
{
    return vector * inversesqrt(max(dot(vector, vector), SquaredLengthEpsilon));
}
bool ReconstructPosition(vec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (pixel + PixelCenter) * Horizon.ViewportRadiusBias.xy;
    // Undo the geometry stage's clip-Y flip for the CPU camera convention.
    vec4 homogeneous = Horizon.InverseViewProjection *
        vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    position = vec3(0.0);
    if (abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return !any(isnan(position)) && !any(isinf(position));
}
bool ProjectRadius(vec3 position, vec3 direction, vec2 centerUv, out vec2 pixelOffset)
{
    vec4 projected = Horizon.ViewProjection *
        vec4(position + direction * Horizon.ViewportRadiusBias.z, 1.0);
    pixelOffset = vec2(0.0);
    if (any(isnan(projected)) || any(isinf(projected)) || projected.w <= HomogeneousEpsilon) return false;
    vec2 ndc = projected.xy / projected.w;
    vec2 uv = vec2(ndc.x * PixelCenter + PixelCenter, PixelCenter - ndc.y * PixelCenter);
    pixelOffset = (uv - centerUv) / Horizon.ViewportRadiusBias.xy;
    return !any(isnan(pixelOffset)) && !any(isinf(pixelOffset));
}

float HorizonVisibility(ivec2 pixel, float depth, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(sampler2D(OpaqueDepth, PointSampler), 0);
    vec2 centerUv = (vec2(pixel) + PixelCenter) * Horizon.ViewportRadiusBias.xy;
    vec3 viewAxis = -Horizon.ForwardIntensity.xyz;
    float radiusSquared = Horizon.ViewportRadiusBias.z * Horizon.ViewportRadiusBias.z;
    float normalView = max(dot(normal, viewAxis), TangentNormalEpsilon);
    // gl_FragCoord has a top-left origin. Hash integer pixels, matching every API.
    float rotation = fract(RotationHashScale * fract(dot(vec2(pixel), RotationHashDirection))) * 2.0 * Pi;
    float blocked = 0.0;
    for (int directionIndex = 0; directionIndex < DirectionCount; ++directionIndex)
    {
        float angle = rotation + 2.0 * Pi * float(directionIndex) / float(DirectionCount);
        vec2 screenDirection = vec2(cos(angle), sin(angle));
        vec3 adjacent;
        if (!ReconstructPosition(vec2(pixel) + screenDirection, depth, adjacent)) continue;
        vec3 direction = adjacent - position;
        direction -= viewAxis * dot(direction, viewAxis);
        float directionSquared = dot(direction, direction);
        if (directionSquared <= SquaredLengthEpsilon) continue;
        direction *= inversesqrt(directionSquared);
        vec2 pixelOffset;
        if (!ProjectRadius(position, direction, centerUv, pixelOffset)) continue;

        float largestHorizon = 0.0;
        ivec2 previousPixel = pixel;
        for (int step = 1; step <= StepsPerDirection; ++step)
        {
            float fraction = float(step) / float(StepsPerDirection);
            vec2 snapped = floor(vec2(pixel) + pixelOffset * fraction + PixelCenter);
            // Check the floating coordinate before converting it to an integer.
            if (any(lessThan(snapped, vec2(0.0))) || any(greaterThanEqual(snapped, vec2(extent)))) continue;
            ivec2 neighbor = ivec2(snapped);
            if (all(equal(neighbor, previousPixel)) || all(equal(neighbor, pixel))) continue;
            previousPixel = neighbor;
            float neighborDepth = DepthAt(neighbor);
            if (neighborDepth >= ClearDepth) continue;
            vec3 neighborPosition;
            if (!ReconstructPosition(vec2(neighbor), neighborDepth, neighborPosition)) continue;
            vec3 delta = neighborPosition - position;
            float distanceSquared = dot(delta, delta);
            if (distanceSquared <= SquaredLengthEpsilon || distanceSquared >= radiusSquared) continue;

            float elevation = dot(delta, viewAxis);
            vec3 actualDirection = delta - viewAxis * elevation;
            float lateralSquared = dot(actualDirection, actualDirection);
            if (lateralSquared <= SquaredLengthEpsilon) continue;
            actualDirection *= inversesqrt(lateralSquared);
            float tangentAngle = atan(-dot(normal, actualDirection), normalView);
            float biasedTangent = clamp(tangentAngle + Horizon.ViewportRadiusBias.w, -HalfPi, HalfPi);
            float sineHorizon = elevation * inversesqrt(distanceSquared);
            float relativeHorizon = max(sineHorizon - sin(biasedTangent), 0.0);
            float newHorizon = max(relativeHorizon - largestHorizon, 0.0);
            blocked += newHorizon * (1.0 - distanceSquared / radiusSquared);
            largestHorizon = max(largestHorizon, relativeHorizon);
        }
    }
    return clamp(1.0 - Horizon.ForwardIntensity.w * blocked / float(DirectionCount), 0.0, 1.0);
}

float FilterVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(sampler2D(OpaqueDepth, PointSampler), 0);
    ivec2 axis = Horizon.Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = Horizon.ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weightedVisibility = 0.0;
    float totalWeight = 0.0;
    for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + axis * offset, ivec2(0), extent - 1);
        float depth = DepthAt(neighbor);
        vec4 neighborNormal = NormalAt(neighbor);
        if (!IsLitSurface(depth, neighborNormal)) continue;
        vec3 neighborPosition;
        if (!ReconstructPosition(vec2(neighbor), depth, neighborPosition)) continue;
        float viewDistance = dot(neighborPosition - position, Horizon.ForwardIntensity.xyz);
        float spatialDistance = float(offset) / SpatialSigma;
        float normalizedDepth = viewDistance / depthSigma;
        float weight = exp(-PixelCenter * (spatialDistance * spatialDistance + normalizedDepth * normalizedDepth));
        weight *= pow(clamp(dot(normal, UnitVector(neighborNormal.xyz)), 0.0, 1.0), NormalWeightExponent);
        weightedVisibility += texelFetch(sampler2D(PreviousVisibility, PointSampler), neighbor, 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > FilterWeightEpsilon ? weightedVisibility / totalWeight : 1.0;
}

void main()
{
    // Preserve an exact identity through both half-float filters and blending.
    if (Horizon.ForwardIntensity.w <= 0.0)
    {
        OutValue = Horizon.Options.x == CompositePass ? vec4(0.0) : vec4(1.0);
        return;
    }
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float depth = DepthAt(pixel);
    vec4 materialNormal = NormalAt(pixel);
    vec3 position = vec3(0.0);
    bool lit = IsLitSurface(depth, materialNormal) && ReconstructPosition(vec2(pixel), depth, position);
    if (Horizon.Options.x == CompositePass)
    {
        float visibility = lit ? clamp(texelFetch(sampler2D(PreviousVisibility, PointSampler), pixel, 0).r, 0.0, 1.0) : 1.0;
        vec3 ambient = texelFetch(sampler2D(AmbientMetallic, PointSampler), pixel, 0).rgb;
        // RGB-only reverse subtraction consumes the loaded HDR target, never a
        // scene-color sample. Direct illumination, emission and alpha survive.
        OutValue = vec4(ambient * (1.0 - visibility), 0.0);
        return;
    }
    float visibility = 1.0;
    if (lit)
    {
        vec3 normal = UnitVector(materialNormal.xyz);
        visibility = Horizon.Options.x == EstimatePass
            ? HorizonVisibility(pixel, depth, position, normal) : FilterVisibility(pixel, position, normal);
    }
    OutValue = vec4(visibility, visibility, visibility, 1.0);
}
#endif
