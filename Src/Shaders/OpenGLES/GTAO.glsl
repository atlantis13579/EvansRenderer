#version 300 es
// Jimenez et al., Practical Realtime Strategies for Accurate Indirect Occlusion,
// ATVI-TR-16-01 (2016), equations 5-8: cosine-weighted horizon integration.
// https://www.activision.com/cdn/research/PracticalRealtimeStrategiesTRfinal.pdf
// Eight spatial slices replace temporal sampling here. Periodic hemisphere
// clipping preserves authored normals, including normals facing away from view.
// Normalizing against the same unoccluded quadrature keeps isolated planes lit.
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
uniform vec4 ViewportRadiusBias; // Inverse extent, world-space radius and height bias.
uniform vec4 ForwardIntensity; // Optical forward for depth filtering, then intensity; local V is reconstructed.
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
const int SliceCount = 8;
const int StepsPerSide = 8;
const int BilateralRadius = 2;
const int HemisphereWrapCount = 1;
const float Pi = 3.141592653589793;
const float HalfPi = Pi * 0.5;
const float PrimitiveScale = 0.25;
const float TexelCenter = 0.5;
const float InteriorClipDepth = 0.5;
const float ClearDepth = 1.0;
const float HomogeneousEpsilon = 1.0e-7;
const float MinimumBaseline = 1.0e-7;
const float RangeFadeStart = 0.5;
const float OpenHorizonCosine = -1.0;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float NormalSimilarityExponent = 8.0;
const float MinimumFilterWeight = 1.0e-7;
const float PixelNoiseScale = 52.9829189;
const vec2 PixelNoiseDirection = vec2(0.06711056, 0.00583715);
// RGBA16F may truncate a component by one ULP; include its subnormal error.
const float HalfNormalRelativeError = 1.0 / 1024.0;
const float HalfNormalSubnormalStep = 1.0 / 16777216.0;

bool FiniteVector(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool NormalizeVector(inout vec3 value)
{
    if (!FiniteVector(value)) return false;
    float scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (scale <= 0.0) return false;
    value /= scale;
    value *= inversesqrt(dot(value, value));
    return true;
}
ivec2 NativePixel(ivec2 canonical)
{
    return ivec2(canonical.x, textureSize(SceneDepth, 0).y - 1 - canonical.y);
}
float DepthAt(ivec2 pixel) { return texelFetch(SceneDepth, NativePixel(pixel), 0).r; }
bool Unproject(vec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (pixel + TexelCenter) * ViewportRadiusBias.xy;
    vec4 world = InverseViewProjection * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    position = vec3(0.0);
    if (any(isnan(world)) || any(isinf(world)) || abs(world.w) < HomogeneousEpsilon) return false;
    position = world.xyz / world.w;
    return FiniteVector(position);
}
bool Surface(ivec2 pixel, out vec3 position, out vec3 normal)
{
    float depth = DepthAt(pixel);
    vec4 stored = texelFetch(NormalRoughness, NativePixel(pixel), 0);
    position = vec3(0.0);
    normal = stored.xyz;
    return depth < ClearDepth && stored.w >= 0.0 && NormalizeVector(normal) && Unproject(vec2(pixel), depth, position);
}

float SignedPrimitive(float theta, float normalView, float normalTangent)
{
    float twiceAngle = 2.0 * theta;
    float integral = PrimitiveScale * (normalView * (1.0 - cos(twiceAngle)) +
        normalTangent * (twiceAngle - sin(twiceAngle)));
    // The sign comes from the spherical Jacobian |sin(theta)|.
    return sign(theta) * integral;
}
float SliceIntegral(float lower, float upper, float normalView, float normalTangent, float normalAngle)
{
    // Dot(N,V) and Dot(N,T) retain the projected normal's length. Clip only
    // positive-cosine intervals, including intervals wrapped across +/-pi.
    float integral = 0.0;
    for (int wrap = -HemisphereWrapCount; wrap <= HemisphereWrapCount; ++wrap)
    {
        float center = normalAngle + 2.0 * Pi * float(wrap);
        float clippedLower = max(lower, center - HalfPi);
        float clippedUpper = min(upper, center + HalfPi);
        if (clippedUpper > clippedLower)
            integral += SignedPrimitive(clippedUpper, normalView, normalTangent) -
                SignedPrimitive(clippedLower, normalView, normalTangent);
    }
    return integral;
}
float HorizonCosine(ivec2 pixel, vec3 position, vec3 normal, vec3 view, vec3 tangent, float side)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    float radius = ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    const float NormalEncodingVectorError = HalfNormalRelativeError + sqrt(3.0) * HalfNormalSubnormalStep;
    const float NormalEncodingSlopeBound = NormalEncodingVectorError / (1.0 - NormalEncodingVectorError);
    vec2 projectedNormal = vec2(dot(normal, view), dot(normal, tangent));
    float projectedNormalSquared = dot(projectedNormal, projectedNormal);
    float horizon = OpenHorizonCosine;
    ivec2 previousPixel = pixel;
    for (int stepIndex = 1; stepIndex <= StepsPerSide; ++stepIndex)
    {
        float fraction = float(stepIndex) / float(StepsPerSide);
        // Quadratic spacing concentrates samples near the center.
        float distanceAlongSlice = radius * fraction * fraction;
        vec4 projected = ViewProjection * vec4(position + side * tangent * distanceAlongSlice, 1.0);
        if (any(isnan(projected)) || any(isinf(projected)) || projected.w <= HomogeneousEpsilon) continue;
        vec2 uv = projected.xy / projected.w * vec2(0.5, -0.5) + 0.5;
        if (any(isnan(uv)) || any(isinf(uv)) || any(lessThan(uv, vec2(0.0))) ||
            any(greaterThanEqual(uv, vec2(1.0)))) continue;
        ivec2 samplePixel = ivec2(floor(uv * vec2(extent)));
        if (all(equal(samplePixel, previousPixel))) continue;
        previousPixel = samplePixel;
        if (all(equal(samplePixel, pixel))) continue;
        float depth = DepthAt(samplePixel);
        vec3 sampled;
        if (depth >= ClearDepth || !Unproject(vec2(samplePixel), depth, sampled)) continue;
        vec3 delta = sampled - position;
        float distanceSquared = dot(delta, delta);
        float tangentComponent = dot(delta, tangent);
        if (!FiniteVector(delta) || distanceSquared >= radiusSquared ||
            side * tangentComponent <= 0.0) continue;
        float worldDistance = sqrt(distanceSquared);
        float worldHeight = dot(normal, delta);
        // A format-derived slope bound prevents quantized normals from occluding their own plane.
        if (worldHeight <= ViewportRadiusBias.w + worldDistance * NormalEncodingSlopeBound) continue;
        vec2 slicePair = vec2(dot(delta, view), tangentComponent);
        // Texel snapping leaves the slice. Restore the sample's actual normal height;
        // the common squared-normal factor cancels in the angle, avoiding division.
        slicePair = slicePair * projectedNormalSquared + projectedNormal *
            (worldHeight - dot(projectedNormal, slicePair));
        if (side * slicePair.y <= 0.0) continue;
        float scale = max(abs(slicePair.x), abs(slicePair.y));
        if (scale <= 0.0 || isnan(scale) || isinf(scale)) continue;
        slicePair /= scale;
        float cosine = slicePair.x * inversesqrt(dot(slicePair, slicePair));
        float weight = 1.0 - smoothstep(radius * RangeFadeStart, radius, worldDistance);
        horizon = max(horizon, mix(OpenHorizonCosine, clamp(cosine, -1.0, 1.0), weight));
    }
    return horizon;
}
float GroundTruthVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    vec3 nearPoint, interiorPoint, adjacent;
    if (!Unproject(vec2(pixel), 0.0, nearPoint) || !Unproject(vec2(pixel), InteriorClipDepth, interiorPoint) ||
        !Unproject(vec2(pixel) + vec2(1.0, 0.0), DepthAt(pixel), adjacent)) return 1.0;
    vec3 view = nearPoint - interiorPoint;
    if (!NormalizeVector(view)) return 1.0;
    vec3 right = adjacent - position;
    right -= view * dot(right, view);
    if (!NormalizeVector(right)) return 1.0;
    vec3 down = cross(right, view);
    // Interleaved-gradient rotation uses integer top-left pixels on every API.
    float rotation = fract(PixelNoiseScale * fract(dot(vec2(pixel), PixelNoiseDirection)));
    float visibleSum = 0.0;
    float baselineSum = 0.0;
    for (int sliceIndex = 0; sliceIndex < SliceCount; ++sliceIndex)
    {
        float phi = (float(sliceIndex) / float(SliceCount) + rotation) * Pi;
        vec3 tangent = cos(phi) * right + sin(phi) * down;
        float normalView = dot(normal, view);
        float normalTangent = dot(normal, tangent);
        if (normalView == 0.0 && normalTangent == 0.0) continue;
        float normalAngle = atan(normalTangent, normalView);
        float negativeCosine = HorizonCosine(pixel, position, normal, view, tangent, -1.0);
        float positiveCosine = HorizonCosine(pixel, position, normal, view, tangent, 1.0);
        float lower = -acos(clamp(negativeCosine, -1.0, 1.0));
        float upper = acos(clamp(positiveCosine, -1.0, 1.0));
        visibleSum += SliceIntegral(lower, upper, normalView, normalTangent, normalAngle);
        baselineSum += SliceIntegral(-Pi, Pi, normalView, normalTangent, normalAngle);
    }
    if (baselineSum <= MinimumBaseline) return 1.0;
    return clamp(1.0 - ForwardIntensity.w * (1.0 - visibleSum / baselineSum), 0.0, 1.0);
}
float FilterVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    ivec2 axis = Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weighted = 0.0;
    float totalWeight = 0.0;
    for (int offset = -BilateralRadius; offset <= BilateralRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + axis * offset, ivec2(0), extent - 1);
        vec3 neighborPosition, neighborNormal;
        if (!Surface(neighbor, neighborPosition, neighborNormal)) continue;
        float depthDelta = dot(neighborPosition - position, ForwardIntensity.xyz);
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialSigma * SpatialSigma));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0 * depthSigma * depthSigma));
        float normalWeight = pow(clamp(dot(normal, neighborNormal), 0.0, 1.0), NormalSimilarityExponent);
        float weight = spatialWeight * depthWeight * normalWeight;
        weighted += texelFetch(PreviousAO, NativePixel(neighbor), 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > MinimumFilterWeight ? clamp(weighted / totalWeight, 0.0, 1.0) : 1.0;
}
void main()
{
    if (ForwardIntensity.w <= 0.0)
    {
        OutColor = Options.x == CompositePass ? vec4(0.0) : vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }
    ivec2 pixel = ivec2(int(gl_FragCoord.x), textureSize(SceneDepth, 0).y - 1 - int(gl_FragCoord.y));
    vec3 position, normal;
    bool valid = Surface(pixel, position, normal);
    if (Options.x == EstimatePass)
        OutColor = vec4(valid ? GroundTruthVisibility(pixel, position, normal) : 1.0, 0.0, 0.0, 1.0);
    else if (Options.x == HorizontalPass || Options.x == VerticalPass)
        OutColor = vec4(valid ? FilterVisibility(pixel, position, normal) : 1.0, 0.0, 0.0, 1.0);
    else
    {
        if (!valid) { OutColor = vec4(0.0); return; }
        ivec2 native = NativePixel(pixel);
        float visibility = clamp(texelFetch(PreviousAO, native, 0).r, 0.0, 1.0);
        vec3 ambient = max(texelFetch(AmbientMetallic, native, 0).rgb, vec3(0.0));
        OutColor = vec4(ambient * (1.0 - visibility), 0.0);
    }
}
#endif
