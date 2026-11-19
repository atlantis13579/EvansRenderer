#version 450
// Independently authored Vulkan GTAO. A view-centered slice integrates the
// cosine-weighted visible arc analytically, including the projected normal's
// length. Jimenez et al., "Practical Realtime Strategies for Accurate Indirect
// Occlusion" (2016), equations 5-8:
// https://www.activision.com/cdn/research/PracticalRealtimeStrategiesTRfinal.pdf
// Periodic hemisphere clipping retains the authored material normal, including
// normals pointing away from the view. Eight full-resolution spatial slices and
// bilateral reconstruction replace the paper's sparse temporal implementation.
// Dividing by the same finite-slice unoccluded integral preserves both empty and
// fully occluded limits. Radius attenuation is restricted to the outer half.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform GroundTruthConstants
{
    mat4 ViewProjection;
    mat4 InverseViewProjection;
    vec4 ViewportRadiusBias;
    vec4 ForwardIntensity;
    uvec4 Options;
} GroundTruth;
layout(set = 0, binding = 1) uniform texture2D OpaqueDepth;
layout(set = 0, binding = 2) uniform texture2D MaterialNormal;
layout(set = 0, binding = 3) uniform texture2D AmbientRadiance;
layout(set = 0, binding = 4) uniform texture2D PreviousVisibility;
layout(set = 0, binding = 5) uniform sampler PointSampler;
layout(location = 0) out vec4 OutValue;

const uint EstimatePass = 0u;
const uint HorizontalPass = 1u;
const uint CompositePass = 3u;
const int SliceCount = 8;
const int StepsPerSide = 8;
const int FilterRadius = 2;
const int PeriodicWindowRadius = 1;
const float Pi = 3.141592653589793;
const float HalfPi = Pi * 0.5;
const float IntegralScale = 0.25;
const float PixelCenter = 0.5;
const float InteriorClipDepth = 0.5;
const float ClearDepth = 1.0;
const float OpenHorizonCosine = -1.0;
const float HomogeneousEpsilon = 1.0e-7;
const float BaselineEpsilon = 1.0e-7;
const float HalfNormalRelativeError = 1.0 / 1024.0;
const float HalfNormalSubnormalStep = 1.0 / 16777216.0;
const float NormalEncodingError = HalfNormalRelativeError + sqrt(3.0) * HalfNormalSubnormalStep;
const float NormalEncodingSlopeBound = NormalEncodingError / (1.0 - NormalEncodingError);
const float FalloffStartRadiusFraction = 0.5;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float NormalWeightExponent = 8.0;
const float FilterWeightEpsilon = 1.0e-7;
const vec2 RotationHashDirection = vec2(0.06711056, 0.00583715);
const float RotationHashScale = 52.9829189;

bool Finite(float value) { return !isnan(value) && !isinf(value); }
bool Finite3(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool NormalizeVector(inout vec3 value)
{
    if (!Finite3(value)) return false;
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (scale <= 0.0) return false;
    value /= scale;
    value *= inversesqrt(dot(value, value));
    return true;
}
float DepthAt(ivec2 pixel)
{
    return texelFetch(sampler2D(OpaqueDepth, PointSampler), pixel, 0).r;
}
bool ValidDepth(float depth)
{
    return Finite(depth) && depth >= 0.0 && depth < ClearDepth;
}
bool ReconstructPosition(vec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (pixel + PixelCenter) * GroundTruth.ViewportRadiusBias.xy;
    // Undo the Vulkan geometry stage's clip-Y flip before inverse projection.
    vec4 homogeneous = GroundTruth.InverseViewProjection *
        vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    position = vec3(0.0);
    if (any(isnan(homogeneous)) || any(isinf(homogeneous)) ||
        abs(homogeneous.w) < HomogeneousEpsilon) return false;
    position = homogeneous.xyz / homogeneous.w;
    return Finite3(position);
}
bool ReadSurface(ivec2 pixel, out float depth, out vec3 position, out vec3 normal)
{
    depth = DepthAt(pixel);
    vec4 material = texelFetch(sampler2D(MaterialNormal, PointSampler), pixel, 0);
    normal = material.xyz;
    position = vec3(0.0);
    return ValidDepth(depth) && material.w >= 0.0 && NormalizeVector(normal) &&
        ReconstructPosition(vec2(pixel), depth, position);
}
bool ProjectSample(vec3 position, ivec2 extent, out ivec2 pixel)
{
    pixel = ivec2(0);
    vec4 projected = GroundTruth.ViewProjection * vec4(position, 1.0);
    if (any(isnan(projected)) || any(isinf(projected)) || projected.w <= HomogeneousEpsilon) return false;
    vec2 ndc = projected.xy / projected.w;
    vec2 uv = vec2(ndc.x * PixelCenter + PixelCenter, PixelCenter - ndc.y * PixelCenter);
    if (any(isnan(uv)) || any(isinf(uv)) || any(lessThan(uv, vec2(0.0))) ||
        any(greaterThanEqual(uv, vec2(1.0)))) return false;
    pixel = ivec2(floor(uv * vec2(extent)));
    return true;
}

float SlicePrimitive(float angle, float normalView, float normalTangent)
{
    // Expanding the projected-normal length times equation 7 avoids normalizing
    // the projection. The sign accounts for the spherical Jacobian |sin(theta)|.
    float doubledAngle = 2.0 * angle;
    float primitive = (normalView * (1.0 - cos(doubledAngle)) +
        normalTangent * (doubledAngle - sin(doubledAngle))) * IntegralScale;
    return sign(angle) * primitive;
}
float VisibleSliceIntegral(float lowerHorizon, float upperHorizon,
    float normalView, float normalTangent, float normalAngle)
{
    float integral = 0.0;
    // A mapped normal can point away from the viewer. Its positive cosine lobe
    // may cross +/-pi, so intersect all relevant periodic copies of the lobe.
    for (int period = -PeriodicWindowRadius; period <= PeriodicWindowRadius; ++period)
    {
        float center = normalAngle + float(period) * 2.0 * Pi;
        float lower = max(lowerHorizon, center - HalfPi);
        float upper = min(upperHorizon, center + HalfPi);
        if (upper > lower)
            integral += SlicePrimitive(upper, normalView, normalTangent) -
                SlicePrimitive(lower, normalView, normalTangent);
    }
    return integral;
}

float SearchHorizon(ivec2 pixel, ivec2 extent, vec3 position, vec3 normal,
    vec3 view, vec3 tangent, float side)
{
    float horizonCosine = OpenHorizonCosine;
    float radius = GroundTruth.ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    vec2 projectedNormal = vec2(dot(normal, view), dot(normal, tangent));
    float projectedNormalSquared = dot(projectedNormal, projectedNormal);
    ivec2 previousPixel = pixel;
    for (int step = 1; step <= StepsPerSide; ++step)
    {
        float fraction = float(step) / float(StepsPerSide);
        float sampleRadius = radius * fraction * fraction;
        ivec2 neighbor;
        // Project each world-space step independently; a distant endpoint may
        // cross the near plane while earlier samples in the same slice remain valid.
        if (!ProjectSample(position + side * tangent * sampleRadius, extent, neighbor)) continue;
        if (all(equal(neighbor, previousPixel)) || all(equal(neighbor, pixel))) continue;
        previousPixel = neighbor;
        float depth = DepthAt(neighbor);
        if (!ValidDepth(depth)) continue;
        vec3 samplePosition;
        if (!ReconstructPosition(vec2(neighbor), depth, samplePosition)) continue;
        vec3 delta = samplePosition - position;
        float distanceSquared = dot(delta, delta);
        if (!Finite(distanceSquared) || distanceSquared >= radiusSquared) continue;
        float distance = sqrt(distanceSquared);
        float worldHeight = dot(normal, delta);
        // RGBA16F conversion can truncate rather than round to nearest. Its
        // relative and subnormal errors bound ambiguous tangent directions;
        // normalization divides the bound by the smallest possible length.
        if (worldHeight <= GroundTruth.ViewportRadiusBias.w + distance * NormalEncodingSlopeBound) continue;
        float tangentDistance = dot(delta, tangent);
        if (side * tangentDistance <= 0.0) continue;
        vec2 sliceDelta = vec2(dot(delta, view), tangentDistance);
        // Snapped samples leave the ideal slice. Preserve their signed height
        // above the material tangent plane when projecting back into that slice.
        // A common positive scale cancels in the horizon angle, avoiding division
        // by a nearly zero projected-normal length.
        sliceDelta = sliceDelta * projectedNormalSquared + projectedNormal *
            (worldHeight - dot(projectedNormal, sliceDelta));
        if (side * sliceDelta.y <= 0.0) continue;
        float scale = max(abs(sliceDelta.x), abs(sliceDelta.y));
        if (!Finite(scale) || scale <= 0.0) continue;
        sliceDelta /= scale;
        float candidate = sliceDelta.x * inversesqrt(dot(sliceDelta, sliceDelta));
        float weight = 1.0 - smoothstep(radius * FalloffStartRadiusFraction, radius, distance);
        candidate = mix(OpenHorizonCosine, clamp(candidate, -1.0, 1.0), weight);
        horizonCosine = max(horizonCosine, candidate);
    }
    return horizonCosine;
}

float GroundTruthVisibility(ivec2 pixel, float depth, vec3 position, vec3 normal)
{
    vec3 nearPosition, interiorPosition, adjacentPosition;
    if (!ReconstructPosition(vec2(pixel), 0.0, nearPosition) ||
        !ReconstructPosition(vec2(pixel), InteriorClipDepth, interiorPosition) ||
        !ReconstructPosition(vec2(pixel) + vec2(1.0, 0.0), depth, adjacentPosition)) return 1.0;
    vec3 view = nearPosition - interiorPosition;
    if (!NormalizeVector(view)) return 1.0;
    vec3 tangentX = adjacentPosition - position;
    tangentX -= view * dot(tangentX, view);
    if (!NormalizeVector(tangentX)) return 1.0;
    vec3 tangentY = cross(tangentX, view);
    ivec2 extent = textureSize(sampler2D(OpaqueDepth, PointSampler), 0);
    // Integer top-left pixels define identical static rotations on every API.
    float rotation = fract(RotationHashScale * fract(dot(vec2(pixel), RotationHashDirection)));
    float visibleSum = 0.0;
    float baselineSum = 0.0;
    for (int slice = 0; slice < SliceCount; ++slice)
    {
        float angle = (float(slice) / float(SliceCount) + rotation) * Pi;
        vec3 tangent = cos(angle) * tangentX + sin(angle) * tangentY;
        float normalView = dot(normal, view);
        float normalTangent = dot(normal, tangent);
        // A zero projection has zero contribution; avoid atan(0, 0).
        if (normalView == 0.0 && normalTangent == 0.0) continue;
        float normalAngle = atan(normalTangent, normalView);
        float negativeCosine = SearchHorizon(pixel, extent, position, normal, view, tangent, -1.0);
        float positiveCosine = SearchHorizon(pixel, extent, position, normal, view, tangent, 1.0);
        visibleSum += VisibleSliceIntegral(-acos(clamp(negativeCosine, -1.0, 1.0)),
            acos(clamp(positiveCosine, -1.0, 1.0)), normalView, normalTangent, normalAngle);
        baselineSum += VisibleSliceIntegral(-Pi, Pi, normalView, normalTangent, normalAngle);
    }
    if (baselineSum <= BaselineEpsilon) return 1.0;
    return clamp(1.0 - GroundTruth.ForwardIntensity.w * (1.0 - visibleSum / baselineSum), 0.0, 1.0);
}

float FilterVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(sampler2D(OpaqueDepth, PointSampler), 0);
    ivec2 axis = GroundTruth.Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = GroundTruth.ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weightedVisibility = 0.0;
    float totalWeight = 0.0;
    for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + axis * offset, ivec2(0), extent - 1);
        float depth;
        vec3 neighborPosition, neighborNormal;
        if (!ReadSurface(neighbor, depth, neighborPosition, neighborNormal)) continue;
        float viewDistance = dot(neighborPosition - position, GroundTruth.ForwardIntensity.xyz);
        float spatialDistance = float(offset) / SpatialSigma;
        float normalizedDepth = viewDistance / depthSigma;
        float weight = exp(-(spatialDistance * spatialDistance + normalizedDepth * normalizedDepth) / 2.0);
        weight *= pow(clamp(dot(normal, neighborNormal), 0.0, 1.0), NormalWeightExponent);
        weightedVisibility += texelFetch(sampler2D(PreviousVisibility, PointSampler), neighbor, 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > FilterWeightEpsilon ? weightedVisibility / totalWeight : 1.0;
}

void main()
{
    // Keep disabled strength exact through floating-point filtering and blending.
    if (GroundTruth.ForwardIntensity.w <= 0.0)
    {
        OutValue = GroundTruth.Options.x == CompositePass ? vec4(0.0) : vec4(1.0);
        return;
    }
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float depth;
    vec3 position, normal;
    bool lit = ReadSurface(pixel, depth, position, normal);
    if (GroundTruth.Options.x == CompositePass)
    {
        float visibility = lit ? clamp(texelFetch(sampler2D(PreviousVisibility, PointSampler), pixel, 0).r, 0.0, 1.0) : 1.0;
        vec3 ambient = texelFetch(sampler2D(AmbientRadiance, PointSampler), pixel, 0).rgb;
        // RGB-only reverse subtraction preserves direct light, emission and alpha.
        OutValue = vec4(ambient * (1.0 - visibility), 0.0);
        return;
    }
    float visibility = 1.0;
    if (lit)
        visibility = GroundTruth.Options.x == EstimatePass
            ? GroundTruthVisibility(pixel, depth, position, normal) : FilterVisibility(pixel, position, normal);
    OutValue = vec4(visibility, visibility, visibility, 1.0);
}
#endif
