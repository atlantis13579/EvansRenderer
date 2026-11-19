#version 300 es
// Mittring, Finding Next Gen - CryEngine 2 (SIGGRAPH 2007), section 8.5.4.3.
// Project a rotated cosine hemisphere into opaque depth; bilateral filtering
// preserves depth/normal boundaries. Composite removes only lost ambient light.
// No noise asset is needed: a deterministic pixel rotation decorrelates samples.
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
const uint HemispherePass = 0u;
const uint HorizontalPass = 1u;
const uint VerticalPass = 2u;
const uint CompositePass = 3u;
const int HemisphereSamples = 24;
const int BilateralRadius = 2;
const float Pi = 3.141592653589793;
const float GoldenAngle = Pi * (3.0 - sqrt(5.0));
const float NormalizeEpsilon = 1.0e-12;
const float HomogeneousEpsilon = 1.0e-7;
const float ParallelAxisThreshold = 0.999;
const float MinimumKernelRadiusFraction = 0.1;
const float RangeFadeStartFraction = 0.5;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float NormalSimilarityExponent = 8.0;
const float MinimumFilterWeight = 1.0e-7;
const float PixelNoiseScale = 52.9829189;
const vec2 PixelNoiseDirection = vec2(0.06711056, 0.00583715);

bool WorldPosition(ivec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (vec2(pixel) + 0.5) * ViewportRadiusBias.xy;
    // ES stores [0,1] window depth after its geometry-stage clip conversion.
    vec4 world = InverseViewProjection * vec4(uv * 2.0 - 1.0, depth, 1.0);
    if (abs(world.w) < HomogeneousEpsilon) { position = vec3(0.0); return false; }
    position = world.xyz / world.w;
    return true;
}

bool Surface(ivec2 pixel, out vec3 position, out vec3 normal)
{
    float depth = texelFetch(SceneDepth, pixel, 0).r;
    vec4 materialNormal = texelFetch(NormalRoughness, pixel, 0);
    float magnitudeSquared = dot(materialNormal.xyz, materialNormal.xyz);
    position = vec3(0.0);
    normal = vec3(0.0, 1.0, 0.0);
    if (depth >= 1.0 || materialNormal.w < 0.0 || magnitudeSquared <= NormalizeEpsilon) return false;
    normal = materialNormal.xyz * inversesqrt(magnitudeSquared);
    return WorldPosition(pixel, depth, position);
}

float HemisphereVisibility(ivec2 pixel, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    // Hash top-left integer pixels consistently with the other native APIs.
    vec2 canonicalPixel = vec2(pixel.x, extent.y - 1 - pixel.y);
    float rotation = 2.0 * Pi * fract(PixelNoiseScale * fract(dot(canonicalPixel, PixelNoiseDirection)));
    vec3 axis = abs(normal.y) < ParallelAxisThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(axis, normal));
    vec3 bitangent = cross(normal, tangent);
    float obstruction = 0.0;
    float radius = ViewportRadiusBias.z;
    for (int index = 0; index < HemisphereSamples; ++index)
    {
        float xi = (float(index) + 0.5) / float(HemisphereSamples);
        float azimuth = rotation + float(index) * GoldenAngle;
        float diskRadius = sqrt(xi);
        vec3 direction = tangent * (cos(azimuth) * diskRadius) +
            bitangent * (sin(azimuth) * diskRadius) + normal * sqrt(1.0 - xi);
        float distanceFraction = mix(MinimumKernelRadiusFraction, 1.0, xi * xi);
        vec3 samplePosition = position + direction * (radius * distanceFraction);
        vec4 projected = ViewProjection * vec4(samplePosition, 1.0);
        if (projected.w <= HomogeneousEpsilon) continue;
        vec3 ndc = projected.xyz / projected.w;
        vec2 sampleUV = ndc.xy * 0.5 + 0.5;
        if (any(lessThan(sampleUV, vec2(0.0))) || any(greaterThanEqual(sampleUV, vec2(1.0))) ||
            ndc.z < 0.0 || ndc.z > 1.0) continue;
        ivec2 samplePixel = ivec2(floor(sampleUV * vec2(extent)));
        float sampleDepth = texelFetch(SceneDepth, samplePixel, 0).r;
        if (sampleDepth >= 1.0) continue;
        vec3 scenePosition;
        if (!WorldPosition(samplePixel, sampleDepth, scenePosition)) continue;
        float separation = dot(samplePosition - scenePosition, ForwardIntensity.xyz);
        float rangeWeight = 1.0 - smoothstep(radius * RangeFadeStartFraction, radius, distance(scenePosition, position));
        if (separation > ViewportRadiusBias.w) obstruction += rangeWeight;
    }
    // Rejected samples stay unoccluded instead of shrinking the denominator.
    return clamp(1.0 - ForwardIntensity.w * obstruction / float(HemisphereSamples), 0.0, 1.0);
}

float BilateralVisibility(ivec2 pixel, vec3 position, vec3 normal)
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
        float spatial = exp(-float(offset * offset) / (2.0 * SpatialSigma * SpatialSigma));
        float depthWeight = exp(-depthDelta * depthDelta / (2.0 * depthSigma * depthSigma));
        float normalWeight = pow(clamp(dot(neighborNormal, normal), 0.0, 1.0), NormalSimilarityExponent);
        float weight = spatial * depthWeight * normalWeight;
        total += texelFetch(PreviousAO, neighbor, 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > MinimumFilterWeight ? clamp(total / totalWeight, 0.0, 1.0) : 1.0;
}

void main()
{
    // Preserve the identity setting exactly through half-float filtering targets.
    if (ForwardIntensity.w <= 0.0)
    {
        OutColor = Options.x == CompositePass ? vec4(0.0) : vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 position, normal;
    bool valid = Surface(pixel, position, normal);
    if (Options.x == HemispherePass)
    {
        float visibility = valid ? HemisphereVisibility(pixel, position, normal) : 1.0;
        OutColor = vec4(visibility, 0.0, 0.0, 1.0);
    }
    else if (Options.x == HorizontalPass || Options.x == VerticalPass)
    {
        float visibility = valid ? BilateralVisibility(pixel, position, normal) : 1.0;
        OutColor = vec4(visibility, 0.0, 0.0, 1.0);
    }
    else
    {
        if (!valid) { OutColor = vec4(0.0); return; }
        float visibility = clamp(texelFetch(PreviousAO, pixel, 0).r, 0.0, 1.0);
        vec3 ambient = max(texelFetch(AmbientMetallic, pixel, 0).rgb, vec3(0.0));
        OutColor = vec4(ambient * (1.0 - visibility), 0.0);
    }
}
#endif
