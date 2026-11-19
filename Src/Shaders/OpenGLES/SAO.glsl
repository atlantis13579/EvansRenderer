#version 300 es
// McGuire, Mara and Luebke, Scalable Ambient Obscurance, HPG 2012.
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Scalable-Ambient-Obscurance/McGuire12SAO.pdf
// Updated author appendix: https://casual-effects.com/research/McGuire2012SAO/McGuire12SAO.pdf
// R32F near-relative linear depth and rotated-grid mips keep nine spiral taps
// cache-friendly. Representative texels retain the selected child's exact ray.
// Edge-aware one-sided depth derivatives replace quad derivatives at silhouettes.
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
const uint DepthPreparePass = 4u;
const uint DepthReducePass = 5u;
const int SpiralSampleCount = 9;
const int SpiralTurnCount = 7;
const int BilateralRadius = 2;
const float Pi = 3.141592653589793;
const float InvalidDepth = -1.0;
const float HomogeneousEpsilon = 1.0e-7;
const float RayDenominatorEpsilon = 1.0e-12;
const float SquaredLengthEpsilon = 1.0e-12;
const float InteriorClipDepth = 0.5;
const float MipRadiusIncrement = 16.0;
const float ObscuranceNormalization = 5.0;
const float ObscuranceDistanceEpsilon = 0.01;
const float ObscuranceSquaredEpsilon = ObscuranceDistanceEpsilon * ObscuranceDistanceEpsilon;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float MinimumFilterWeight = 1.0e-7;
const float PixelNoiseScale = 52.9829189;
const vec2 PixelNoiseDirection = vec2(0.06711056, 0.00583715);

bool FiniteVector(vec3 value) { return !any(isnan(value)) && !any(isinf(value)); }
bool ValidDepth(float value) { return value >= 0.0 && !isnan(value) && !isinf(value); }
bool NormalizeVector(inout vec3 value)
{
    if (!FiniteVector(value)) return false;
    float scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (scale <= 0.0) return false;
    value /= scale;
    value *= inversesqrt(dot(value, value));
    return true;
}
ivec2 NativePixel(ivec2 canonical, int level)
{
    return ivec2(canonical.x, textureSize(SceneDepth, level).y - 1 - canonical.y);
}
float DepthAt(ivec2 canonical, int level)
{
    return texelFetch(SceneDepth, NativePixel(canonical, level), level).r;
}
bool InBounds(ivec2 pixel)
{
    return all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, textureSize(SceneDepth, 0)));
}
bool IsLit(ivec2 pixel)
{
    return InBounds(pixel) && texelFetch(NormalRoughness, NativePixel(pixel, 0), 0).w >= 0.0;
}
bool Unproject(vec2 pixel, float clipDepth, out vec3 world)
{
    vec2 uv = (pixel + 0.5) * ViewportRadiusBias.xy;
    vec4 homogeneous = InverseViewProjection * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, clipDepth, 1.0);
    world = vec3(0.0);
    if (abs(homogeneous.w) < HomogeneousEpsilon) return false;
    world = homogeneous.xyz / homogeneous.w;
    return FiniteVector(world);
}
bool WorldFromLinearDepth(vec2 pixel, float depth, out vec3 world)
{
    world = vec3(0.0);
    if (!ValidDepth(depth)) return false;
    vec3 nearPoint, middlePoint;
    if (!Unproject(pixel, 0.0, nearPoint) || !Unproject(pixel, InteriorClipDepth, middlePoint)) return false;
    vec3 ray = middlePoint - nearPoint;
    float denominator = dot(ray, ForwardIntensity.xyz);
    if (abs(denominator) <= RayDenominatorEpsilon || isnan(denominator) || isinf(denominator)) return false;
    world = nearPoint + ray * (depth / denominator);
    return FiniteVector(world);
}
float PrepareLinearDepth(ivec2 pixel)
{
    float hardwareDepth = DepthAt(pixel, 0);
    vec3 world, nearPoint;
    if (hardwareDepth >= 1.0 || !Unproject(vec2(pixel), hardwareDepth, world) || !Unproject(vec2(pixel), 0.0, nearPoint))
        return InvalidDepth;
    // Referencing the near plane avoids requiring a finite far plane or camera origin.
    float depth = dot(world - nearPoint, ForwardIntensity.xyz);
    return isnan(depth) || isinf(depth) ? InvalidDepth : max(depth, 0.0);
}
ivec2 SelectedChild(ivec2 parent, int sourceLevel)
{
    ivec2 parity = ivec2((parent.y & 1) ^ 1, (parent.x & 1) ^ 1);
    return clamp(parent * 2 + parity, ivec2(0), textureSize(SceneDepth, sourceLevel) - 1);
}
ivec2 RepresentativePixel(ivec2 cell, int level)
{
    // Recover the same rotated-grid child at every level, including odd extents.
    for (int sourceLevel = level - 1; sourceLevel >= 0; --sourceLevel)
        cell = SelectedChild(cell, sourceLevel);
    return cell;
}
bool DepthSurface(ivec2 pixel, out float depth, out vec3 position)
{
    depth = InvalidDepth;
    position = vec3(0.0);
    if (!InBounds(pixel)) return false;
    depth = DepthAt(pixel, 0);
    return WorldFromLinearDepth(vec2(pixel), depth, position);
}
bool Surface(ivec2 pixel, out float depth, out vec3 position)
{
    depth = InvalidDepth;
    position = vec3(0.0);
    return IsLit(pixel) && DepthSurface(pixel, depth, position);
}
bool GeometricNormal(ivec2 pixel, float depth, vec3 position, out vec3 normal)
{
    float leftZ, rightZ, upZ, downZ;
    vec3 left, right, up, down;
    // Wire markers disable AO shading, but do not remove the underlying geometry.
    bool leftValid = DepthSurface(pixel + ivec2(-1, 0), leftZ, left);
    bool rightValid = DepthSurface(pixel + ivec2(1, 0), rightZ, right);
    bool upValid = DepthSurface(pixel + ivec2(0, -1), upZ, up);
    bool downValid = DepthSurface(pixel + ivec2(0, 1), downZ, down);
    normal = vec3(0.0);
    if ((!leftValid && !rightValid) || (!upValid && !downValid)) return false;
    bool chooseRight = rightValid && (!leftValid || abs(rightZ - depth) < abs(leftZ - depth));
    bool chooseDown = downValid && (!upValid || abs(downZ - depth) < abs(upZ - depth));
    vec3 horizontal = chooseRight ? right - position : position - left;
    vec3 vertical = chooseDown ? down - position : position - up;
    // Normalize before crossing: validity depends on angle, not world pixel area.
    if (!NormalizeVector(horizontal) || !NormalizeVector(vertical)) return false;
    normal = cross(vertical, horizontal);
    float normalSquared = dot(normal, normal);
    if (normalSquared <= SquaredLengthEpsilon || !NormalizeVector(normal)) return false;
    if (dot(normal, -ForwardIntensity.xyz) < 0.0) normal = -normal;
    return FiniteVector(normal);
}
float EstimateObscurance(ivec2 pixel, float depth, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    float rotation = 2.0 * Pi * fract(PixelNoiseScale * fract(dot(vec2(pixel), PixelNoiseDirection)));
    float radius = ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float sum = 0.0;
    for (int sampleIndex = 0; sampleIndex < SpiralSampleCount; ++sampleIndex)
    {
        float alpha = (float(sampleIndex) + 0.5) / float(SpiralSampleCount);
        float angle = 2.0 * Pi * float(SpiralTurnCount) * alpha + rotation;
        vec3 adjacent;
        if (!WorldFromLinearDepth(vec2(pixel) + vec2(cos(angle), sin(angle)), depth, adjacent)) continue;
        vec3 direction = adjacent - position;
        direction -= ForwardIntensity.xyz * dot(direction, ForwardIntensity.xyz);
        if (!NormalizeVector(direction)) continue;
        vec4 endpoint = ViewProjection * vec4(position + radius * direction, 1.0);
        if (any(isnan(endpoint)) || any(isinf(endpoint)) || endpoint.w <= HomogeneousEpsilon) continue;
        vec2 uv = endpoint.xy / endpoint.w * vec2(0.5, -0.5) + 0.5;
        vec2 tapOffset = (uv * vec2(extent) - (vec2(pixel) + 0.5)) * alpha;
        vec2 snapped = floor(vec2(pixel) + tapOffset + 0.5);
        if (any(isnan(snapped)) || any(isinf(snapped)) || any(lessThan(snapped, vec2(0.0))) ||
            any(greaterThanEqual(snapped, vec2(extent)))) continue;
        int mip = int(clamp(floor(log2(max(length(tapOffset), 1.0) / MipRadiusIncrement)), 0.0, float(Options.z)));
        ivec2 cell = clamp(ivec2(snapped) >> mip, ivec2(0), textureSize(SceneDepth, mip) - 1);
        float sampleDepth = DepthAt(cell, mip);
        ivec2 representative = RepresentativePixel(cell, mip);
        vec3 samplePosition;
        if (!WorldFromLinearDepth(vec2(representative), sampleDepth, samplePosition)) continue;
        vec3 displacement = samplePosition - position;
        float distanceSquared = dot(displacement, displacement);
        if (distanceSquared >= radiusSquared) continue;
        float falloff = max(1.0 - distanceSquared / radiusSquared, 0.0);
        float facing = max(dot(displacement, normal) - ViewportRadiusBias.w, 0.0);
        sum += falloff * falloff * falloff * facing / (distanceSquared + ObscuranceSquaredEpsilon);
    }
    return clamp(1.0 - ObscuranceNormalization * ForwardIntensity.w * sum / float(SpiralSampleCount), 0.0, 1.0);
}
float FilterVisibility(ivec2 pixel, float depth)
{
    ivec2 extent = textureSize(SceneDepth, 0);
    ivec2 direction = Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float weighted = 0.0;
    float totalWeight = 0.0;
    for (int offset = -BilateralRadius; offset <= BilateralRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + direction * offset, ivec2(0), extent - 1);
        if (!IsLit(neighbor)) continue;
        float neighborDepth = DepthAt(neighbor, 0);
        if (!ValidDepth(neighborDepth)) continue;
        float difference = neighborDepth - depth;
        float spatialWeight = exp(-float(offset * offset) / (2.0 * SpatialSigma * SpatialSigma));
        float depthWeight = exp(-difference * difference / (2.0 * depthSigma * depthSigma));
        float weight = spatialWeight * depthWeight;
        weighted += texelFetch(PreviousAO, NativePixel(neighbor, 0), 0).r * weight;
        totalWeight += weight;
    }
    return totalWeight > MinimumFilterWeight ? clamp(weighted / totalWeight, 0.0, 1.0) : 1.0;
}
void main()
{
    int destinationLevel = Options.x == DepthReducePass ? int(Options.y) : 0;
    ivec2 extent = textureSize(SceneDepth, destinationLevel);
    ivec2 pixel = ivec2(int(gl_FragCoord.x), extent.y - 1 - int(gl_FragCoord.y));
    if (Options.x == DepthPreparePass)
    {
        OutColor = vec4(PrepareLinearDepth(pixel), 0.0, 0.0, 1.0);
        return;
    }
    if (Options.x == DepthReducePass)
    {
        int sourceLevel = destinationLevel - 1;
        OutColor = vec4(DepthAt(SelectedChild(pixel, sourceLevel), sourceLevel), 0.0, 0.0, 1.0);
        return;
    }
    if (ForwardIntensity.w <= 0.0)
    {
        OutColor = Options.x == CompositePass ? vec4(0.0) : vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }
    float depth;
    vec3 position;
    bool valid = Surface(pixel, depth, position);
    if (Options.x == EstimatePass)
    {
        vec3 normal;
        float visibility = valid && GeometricNormal(pixel, depth, position, normal)
            ? EstimateObscurance(pixel, depth, position, normal) : 1.0;
        OutColor = vec4(visibility, 0.0, 0.0, 1.0);
    }
    else if (Options.x == HorizontalPass || Options.x == VerticalPass)
        OutColor = vec4(valid ? FilterVisibility(pixel, depth) : 1.0, 0.0, 0.0, 1.0);
    else
    {
        if (!valid) { OutColor = vec4(0.0); return; }
        ivec2 native = NativePixel(pixel, 0);
        float visibility = clamp(texelFetch(PreviousAO, native, 0).r, 0.0, 1.0);
        vec3 ambient = max(texelFetch(AmbientMetallic, native, 0).rgb, vec3(0.0));
        OutColor = vec4(ambient * (1.0 - visibility), 0.0);
    }
}
#endif
