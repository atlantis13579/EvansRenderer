#version 450
// Independent Vulkan Scalable Ambient Obscurance, McGuire, Mara and Luebke (2012).
// https://research.nvidia.com/sites/default/files/pubs/2012-06_Scalable-Ambient-Obscurance/McGuire12SAO.pdf
// Updated falloff: https://casual-effects.com/research/McGuire2012SAO/McGuire12SAO.pdf
// The real R32F hierarchy stores optical distance from the near plane. Rotated
// grid reduction keeps actual samples; reconstruction recovers their original
// coordinates, including clamped odd-sized levels. Bounded, edge-aware depth
// differences replace quad derivatives for geometric normals. The estimator
// uses the Appendix's cubic falloff with unit contrast; the final two filters
// use linear-depth differences and compose only the ambient lighting term.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140, row_major) uniform ScalableConstants
{
    mat4 ViewProjection;
    mat4 InverseViewProjection;
    vec4 ViewportRadiusBias;
    vec4 ForwardIntensity;
    uvec4 Options;
} Scalable;
layout(set = 0, binding = 1) uniform texture2D DepthInput;
layout(set = 0, binding = 2) uniform texture2D MaterialNormal;
layout(set = 0, binding = 3) uniform texture2D AmbientRadiance;
layout(set = 0, binding = 4) uniform texture2D PreviousVisibility;
layout(set = 0, binding = 5) uniform sampler PointSampler;
layout(location = 0) out vec4 OutValue;

const uint EstimatePass = 0u;
const uint HorizontalPass = 1u;
const uint CompositePass = 3u;
const uint PrepareDepthPass = 4u;
const uint ReduceDepthPass = 5u;
const int SampleCount = 9;
const float SpiralTurns = 7.0;
const float MipRadiusIncrement = 16.0;
const float ObscuranceNormalization = 5.0;
const float DistanceRegularization = 0.01;
const float EpsilonSquared = DistanceRegularization * DistanceRegularization;
const int FilterRadius = 2;
const float SpatialSigma = 1.5;
const float DepthSigmaRadiusFraction = 0.25;
const float MinimumFilterWeight = 1.0e-7;
const float HomogeneousEpsilon = 1.0e-7;
const float RayProjectionEpsilon = 1.0e-12;
const float SquaredVectorEpsilon = 1.0e-12;
const float ClearHardwareDepth = 1.0;
const float InvalidLinearDepth = -1.0;
const float NearClipDepth = 0.0;
const float InteriorClipDepth = 0.5;
const float PixelCenter = 0.5;
const float Pi = 3.141592653589793;
const vec2 RotationHashDirection = vec2(0.06711056, 0.00583715);
const float RotationHashScale = 52.9829189;

bool Finite(vec3 value)
{
    return !any(isnan(value)) && !any(isinf(value));
}
bool InBounds(ivec2 pixel, ivec2 extent)
{
    return all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, extent));
}
bool UnitDirection(vec3 value, out vec3 direction)
{
    direction = vec3(0.0);
    if (!Finite(value)) return false;
    float scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (scale <= 0.0) return false;
    value /= scale;
    direction = value * inversesqrt(dot(value, value));
    return true;
}
bool Unproject(vec2 pixel, float depth, out vec3 position)
{
    vec2 uv = (pixel + PixelCenter) * Scalable.ViewportRadiusBias.xy;
    // Vulkan geometry flips clip Y; inverse VP uses the canonical top-left image.
    vec4 world = Scalable.InverseViewProjection *
        vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    position = vec3(0.0);
    if (abs(world.w) < HomogeneousEpsilon) return false;
    position = world.xyz / world.w;
    return Finite(position);
}
float LinearDepthAt(ivec2 pixel, int level)
{
    return texelFetch(sampler2D(DepthInput, PointSampler), pixel, level).r;
}
bool ValidLinearDepth(float depth)
{
    return depth >= 0.0 && !isinf(depth) && !isnan(depth);
}
bool IsLit(ivec2 pixel)
{
    // Material normals are used only for the unlit wireframe marker.
    return texelFetch(sampler2D(MaterialNormal, PointSampler), pixel, 0).w >= 0.0;
}
bool ReconstructWorld(vec2 pixel, float linearDepth, out vec3 position)
{
    position = vec3(0.0);
    vec3 nearPosition, interiorPosition;
    if (!ValidLinearDepth(linearDepth) || !Unproject(pixel, NearClipDepth, nearPosition) ||
        !Unproject(pixel, InteriorClipDepth, interiorPosition)) return false;
    vec3 ray = interiorPosition - nearPosition;
    float projection = dot(ray, Scalable.ForwardIntensity.xyz);
    if (abs(projection) <= RayProjectionEpsilon) return false;
    position = nearPosition + ray * (linearDepth / projection);
    return Finite(position);
}
float PrepareDepth(ivec2 pixel)
{
    float depth = LinearDepthAt(pixel, 0);
    vec3 position, nearPosition;
    if (depth >= ClearHardwareDepth || !Unproject(vec2(pixel), depth, position) ||
        !Unproject(vec2(pixel), NearClipDepth, nearPosition)) return InvalidLinearDepth;
    float linearDepth = dot(position - nearPosition, Scalable.ForwardIntensity.xyz);
    return isnan(linearDepth) || isinf(linearDepth) ? InvalidLinearDepth : max(linearDepth, 0.0);
}
ivec2 RotatedChild(ivec2 parent, ivec2 childExtent)
{
    ivec2 corner = ivec2((parent.y & 1) ^ 1, (parent.x & 1) ^ 1);
    return clamp(parent * 2 + corner, ivec2(0), childExtent - 1);
}
float ReduceDepth(ivec2 pixel)
{
    int sourceMip = int(Scalable.Options.y) - 1;
    ivec2 sourceExtent = textureSize(sampler2D(DepthInput, PointSampler), sourceMip);
    return LinearDepthAt(RotatedChild(pixel, sourceExtent), sourceMip);
}
ivec2 RepresentativePixel(ivec2 coordinate, int level)
{
    // Mirror every reduction decision rather than reconstructing at a mip center.
    for (int childLevel = level - 1; childLevel >= 0; --childLevel)
        coordinate = RotatedChild(coordinate, textureSize(sampler2D(DepthInput, PointSampler), childLevel));
    return coordinate;
}
bool AxisDerivative(ivec2 pixel, ivec2 axis, float centerDepth, vec3 center, out vec3 derivative)
{
    ivec2 extent = textureSize(sampler2D(DepthInput, PointSampler), 0);
    ivec2 before = pixel - axis;
    ivec2 after = pixel + axis;
    float beforeDepth = InBounds(before, extent) ? LinearDepthAt(before, 0) : InvalidLinearDepth;
    float afterDepth = InBounds(after, extent) ? LinearDepthAt(after, 0) : InvalidLinearDepth;
    bool beforeValid = ValidLinearDepth(beforeDepth);
    bool afterValid = ValidLinearDepth(afterDepth);
    derivative = vec3(0.0);
    if (!beforeValid && !afterValid) return false;
    bool useBefore = beforeValid && (!afterValid || abs(beforeDepth - centerDepth) <= abs(afterDepth - centerDepth));
    vec3 neighbor;
    if (!ReconstructWorld(vec2(useBefore ? before : after), useBefore ? beforeDepth : afterDepth, neighbor)) return false;
    // Both branches retain the canonical right/down orientation.
    return UnitDirection(useBefore ? center - neighbor : neighbor - center, derivative);
}
bool GeometricNormal(ivec2 pixel, float depth, vec3 position, out vec3 normal)
{
    normal = vec3(0.0);
    vec3 right, down;
    if (!AxisDerivative(pixel, ivec2(1, 0), depth, position, right) ||
        !AxisDerivative(pixel, ivec2(0, 1), depth, position, down)) return false;
    vec3 perpendicular = cross(down, right);
    if (dot(perpendicular, perpendicular) <= SquaredVectorEpsilon || !UnitDirection(perpendicular, normal)) return false;
    if (dot(normal, -Scalable.ForwardIntensity.xyz) < 0.0) normal = -normal;
    return true;
}

float EstimateVisibility(ivec2 pixel, float depth, vec3 position, vec3 normal)
{
    ivec2 extent = textureSize(sampler2D(DepthInput, PointSampler), 0);
    vec2 centerUv = (vec2(pixel) + PixelCenter) * Scalable.ViewportRadiusBias.xy;
    float radius = Scalable.ViewportRadiusBias.z;
    float radiusSquared = radius * radius;
    float rotation = fract(RotationHashScale * fract(dot(vec2(pixel), RotationHashDirection))) * 2.0 * Pi;
    float sum = 0.0;
    for (int index = 0; index < SampleCount; ++index)
    {
        float alpha = (float(index) + PixelCenter) / float(SampleCount);
        float angle = 2.0 * Pi * SpiralTurns * alpha + rotation;
        vec3 adjacent, direction;
        if (!ReconstructWorld(vec2(pixel) + vec2(cos(angle), sin(angle)), depth, adjacent)) continue;
        vec3 delta = adjacent - position;
        delta -= Scalable.ForwardIntensity.xyz * dot(delta, Scalable.ForwardIntensity.xyz);
        if (!UnitDirection(delta, direction)) continue;
        vec4 endpoint = Scalable.ViewProjection * vec4(position + radius * direction, 1.0);
        if (endpoint.w <= HomogeneousEpsilon || any(isnan(endpoint)) || any(isinf(endpoint))) continue;
        vec2 ndc = endpoint.xy / endpoint.w;
        vec2 uv = vec2(ndc.x * PixelCenter + PixelCenter, PixelCenter - ndc.y * PixelCenter);
        vec2 tapOffset = ((uv - centerUv) / Scalable.ViewportRadiusBias.xy) * alpha;
        vec2 snapped = floor(vec2(pixel) + tapOffset + PixelCenter);
        if (any(isnan(snapped)) || any(lessThan(snapped, vec2(0.0))) ||
            any(greaterThanEqual(snapped, vec2(extent)))) continue;
        float pixelRadius = length(tapOffset);
        int level = int(clamp(floor(log2(max(pixelRadius, 1.0) / MipRadiusIncrement)), 0.0, float(Scalable.Options.z)));
        ivec2 mipExtent = textureSize(sampler2D(DepthInput, PointSampler), level);
        ivec2 coordinate = clamp(ivec2(snapped) >> level, ivec2(0), mipExtent - 1);
        float sampleDepth = LinearDepthAt(coordinate, level);
        if (!ValidLinearDepth(sampleDepth)) continue;
        ivec2 representative = RepresentativePixel(coordinate, level);
        vec3 samplePosition;
        if (!ReconstructWorld(vec2(representative), sampleDepth, samplePosition)) continue;
        vec3 displacement = samplePosition - position;
        float distanceSquared = dot(displacement, displacement);
        if (distanceSquared >= radiusSquared) continue;
        float falloff = max(1.0 - distanceSquared / radiusSquared, 0.0);
        float aboveTangent = max(dot(displacement, normal) - Scalable.ViewportRadiusBias.w, 0.0);
        sum += falloff * falloff * falloff * aboveTangent / (distanceSquared + EpsilonSquared);
    }
    return clamp(1.0 - ObscuranceNormalization * Scalable.ForwardIntensity.w * sum / float(SampleCount), 0.0, 1.0);
}
float FilterVisibility(ivec2 pixel, float depth)
{
    ivec2 extent = textureSize(sampler2D(DepthInput, PointSampler), 0);
    ivec2 axis = Scalable.Options.x == HorizontalPass ? ivec2(1, 0) : ivec2(0, 1);
    float depthSigma = Scalable.ViewportRadiusBias.z * DepthSigmaRadiusFraction;
    float numerator = 0.0;
    float denominator = 0.0;
    for (int offset = -FilterRadius; offset <= FilterRadius; ++offset)
    {
        ivec2 neighbor = clamp(pixel + offset * axis, ivec2(0), extent - 1);
        float neighborDepth = LinearDepthAt(neighbor, 0);
        if (!ValidLinearDepth(neighborDepth) || !IsLit(neighbor)) continue;
        float spatialDistance = float(offset) / SpatialSigma;
        float depthDistance = (neighborDepth - depth) / depthSigma;
        float weight = exp(-PixelCenter * (spatialDistance * spatialDistance + depthDistance * depthDistance));
        numerator += weight * texelFetch(sampler2D(PreviousVisibility, PointSampler), neighbor, 0).r;
        denominator += weight;
    }
    return denominator > MinimumFilterWeight ? numerator / denominator : 1.0;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Scalable.Options.x == PrepareDepthPass)
    {
        OutValue = vec4(PrepareDepth(pixel), 0.0, 0.0, 1.0);
        return;
    }
    if (Scalable.Options.x == ReduceDepthPass)
    {
        OutValue = vec4(ReduceDepth(pixel), 0.0, 0.0, 1.0);
        return;
    }
    // The hierarchy is still built at zero strength; all visibility/composition
    // stages preserve exact identity through half-float storage and blending.
    if (Scalable.ForwardIntensity.w <= 0.0)
    {
        OutValue = Scalable.Options.x == CompositePass ? vec4(0.0) : vec4(1.0);
        return;
    }
    float depth = LinearDepthAt(pixel, 0);
    bool valid = ValidLinearDepth(depth) && IsLit(pixel);
    if (Scalable.Options.x == CompositePass)
    {
        float visibility = valid ? clamp(texelFetch(sampler2D(PreviousVisibility, PointSampler), pixel, 0).r, 0.0, 1.0) : 1.0;
        vec3 ambient = texelFetch(sampler2D(AmbientRadiance, PointSampler), pixel, 0).rgb;
        OutValue = vec4(ambient * (1.0 - visibility), 0.0);
        return;
    }
    float visibility = 1.0;
    if (valid)
    {
        if (Scalable.Options.x == EstimatePass)
        {
            vec3 position, normal;
            if (ReconstructWorld(vec2(pixel), depth, position) && GeometricNormal(pixel, depth, position, normal))
                visibility = EstimateVisibility(pixel, depth, position, normal);
        }
        else visibility = FilterVisibility(pixel, depth);
    }
    OutValue = vec4(visibility, visibility, visibility, 1.0);
}
#endif
