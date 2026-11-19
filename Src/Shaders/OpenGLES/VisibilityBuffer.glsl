#version 310 es
precision highp float;
precision highp int;
// Independent ES 3.1 vertex pulling and material resolve. Vertex/fragment SSBO
// capacity and integer/float MRT renderability must be probed before activation.
// Burns and Hunt (2013): https://jcgt.org/published/0002/02/04/
// Texture gradients are analytical; helper pixels never read another primitive.
// Preserve Toon material color/detail before Deferred applies the light ramp.
// The analytic texture footprint filters RGB while keeping center alpha.
precision highp float;
precision highp int;
const float ToonShadingModel = 4.0;
const float ToonAlbedoExponent = 0.8;
const float ToonPaletteSteps = 16.0;
const float ToonColorSaturation = 1.12;
const float ToonNormalStrength = 0.12;
const vec3 ToonLuma = vec3(0.2126, 0.7152, 0.0722);
const vec3 ToonSkyFill = vec3(0.24, 0.29, 0.4);
vec3 ToonAlbedo(vec3 value, float model)
{
    if (model != ToonShadingModel) return value;
    vec3 paint = pow(clamp(value, 0.0, 1.0), vec3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return clamp(mix(vec3(dot(paint, ToonLuma)), paint, ToonColorSaturation), 0.0, 1.0);
}
vec3 ToonAmbient(vec3 albedo, vec3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
// Average RGB over a screen footprint; center alpha keeps cutout coverage.
// Explicit gradients allow this helper to coexist with vertex entry points.
const float ToonTextureRadius = 1.25;
const int ToonTextureKernelRadius = 1;
const int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
const float ToonTextureSampleWeight = 1.0 / float(ToonTextureKernelWidth * ToonTextureKernelWidth);
vec3 ToonTextureColor(sampler2D image, vec2 uv, vec2 uvDx, vec2 uvDy)
{
    vec3 color = vec3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += textureGrad(image, uv + ToonTextureRadius * (float(column) * uvDx + float(row) * uvDy), uvDx, uvDy).rgb;
    return color * ToonTextureSampleWeight;
}

const uint BackFacing = 0x80000000u, TriangleReferenceMask = 0x7fffffffu;
const uint VerticesPerTriangle = 3u, FlatShading = 1u, PbrShading = 3u;
const float MatrixTolerance = 1.0e-12, VectorTolerance = 1.0e-12, UVTolerance = 1.0e-10;
const float MinimumRoughness = 0.045, MaximumRadiance = 65504.0;
const float ParallelTangentLimit = 0.999;
struct TriangleRecord
{
    vec4 Position0, Position1, Position2;
    vec4 Normal0, Normal1, Normal2;
    vec4 Color0, Color1, Color2;
    vec4 UV01; vec4 UV2; uvec4 MaterialSource;
};
layout(std140, binding = 0) uniform VisibilityFrame
{
    vec4 CameraRows[4];
    vec4 Extent, Tint, DiffuseMaterial, AmbientMaterial, SpecularMaterial, EmissionMaterial;
    vec4 MaterialParameters, AmbientIllumination, SurfaceParameters;
    uvec4 TextureFlags, DrawRange, MaterialTiles;
};
// Raw unsigned vectors avoid aggregate SSBO loads of nested arrays on ANGLE.
// The twelve 16-byte words retain the common 192-byte triangle ABI exactly.
#if !defined(EVANS_VERTEX_STAGE) || !defined(EVANS_VISIBILITY_RESOLVE)
layout(std430, binding = 0) readonly buffer TriangleStorage { uvec4 TriangleWords[]; };
// Separate blocks retain global addresses beyond the per-block device limit.
#if defined(EVANS_VISIBILITY_BANK_WORDS)
layout(std430,binding=4) readonly buffer GeometryOverflow { uvec4 overflowWords[]; };
#endif
uvec4 readGeometryWord(uint word)
{
#if defined(EVANS_VISIBILITY_BANK_WORDS)
    const uint BankWords=EVANS_VISIBILITY_BANK_WORDS / 4u;
    if (word>=BankWords) return overflowWords[word-BankWords];
#endif
    return TriangleWords[word];
}
#endif
#if defined(EVANS_MATERIAL_TILES)
#if defined(EVANS_VERTEX_STAGE)
// A sampled integer list avoids sharing ANGLE's graphics UAV register bank
// between vertex pulling and the fragment material's geometry/map buffers.
uniform highp usampler2D MaterialScheduleImage;
uint MaterialTileWord(uint address)
{
    uint width=uint(textureSize(MaterialScheduleImage,0).x);
    return texelFetch(MaterialScheduleImage,ivec2(address%width,address/width),0).r;
}
#else
layout(std430,binding=1) readonly buffer MaterialMapping { uint MaterialBins[]; };
#endif
#endif
const uint TriangleWordCount = 12u, NormalWord = 3u, ColorWord = 6u, UVWord = 9u, SourceWord = 11u;
#if !defined(EVANS_VERTEX_STAGE) || !defined(EVANS_VISIBILITY_RESOLVE)
TriangleRecord ReadTriangle(uint index)
{
    uint base = index * TriangleWordCount;
    TriangleRecord result;
    result.Position0 = uintBitsToFloat(readGeometryWord(base));
    result.Position1 = uintBitsToFloat(readGeometryWord(base + 1u));
    result.Position2 = uintBitsToFloat(readGeometryWord(base + 2u));
    result.Normal0 = uintBitsToFloat(readGeometryWord(base + NormalWord));
    result.Normal1 = uintBitsToFloat(readGeometryWord(base + NormalWord + 1u));
    result.Normal2 = uintBitsToFloat(readGeometryWord(base + NormalWord + 2u));
    result.Color0 = uintBitsToFloat(readGeometryWord(base + ColorWord));
    result.Color1 = uintBitsToFloat(readGeometryWord(base + ColorWord + 1u));
    result.Color2 = uintBitsToFloat(readGeometryWord(base + ColorWord + 2u));
    result.UV01 = uintBitsToFloat(readGeometryWord(base + UVWord));
    result.UV2 = uintBitsToFloat(readGeometryWord(base + UVWord + 1u));
    result.MaterialSource = readGeometryWord(base + SourceWord);
    return result;
}
#endif
vec4 PositionAt(TriangleRecord data, uint corner)
{ return corner == 0u ? data.Position0 : corner == 1u ? data.Position1 : data.Position2; }
vec4 ColorAt(TriangleRecord data, uint corner)
{ return corner == 0u ? data.Color0 : corner == 1u ? data.Color1 : data.Color2; }
uniform highp usampler2D VisibilityImage;
uniform highp sampler2D BarycentricImage;
uniform highp sampler2D AlbedoImage;
uniform highp sampler2D NormalImage;
uniform highp sampler2D OpacityImage;
vec4 ClipPosition(vec4 point)
{
    // Use the same matrix-vector operation as opaque and motion vertices;
    // separate dot products may contract differently in the native compiler.
    return transpose(mat4(CameraRows[0], CameraRows[1], CameraRows[2], CameraRows[3])) * point;
}
vec2 TextureCoordinate(TriangleRecord triangle, uint corner)
{
    vec2 source = corner == 0u ? triangle.UV01.xy : corner == 1u ? triangle.UV01.zw : triangle.UV2.xy;
    return vec2(source.x, 1.0 - source.y);
}
vec3 NormalizeOr(vec3 value, vec3 fallback)
{ return dot(value, value) > VectorTolerance ? normalize(value) : fallback; }
vec3 WireColor(vec3 value)
{
    const float Threshold = 0.04045, LinearScale = 12.92, PowerScale = 1.055, Offset = 0.055, Exponent = 2.4;
    value = clamp(value, 0.0, 1.0); vec3 power = pow((value + Offset) / PowerScale, vec3(Exponent));
    return mix(power, value / LinearScale, lessThanEqual(value, vec3(Threshold)));
}
#if defined(EVANS_VERTEX_STAGE)
uniform highp int WireFaceSign;
// Preserve depth agreement with the ordinary mesh velocity replay.
invariant gl_Position;
#if !defined(EVANS_VISIBILITY_RESOLVE)
out vec2 Weights;
out vec2 UV;
out float VertexOpacity;
flat out uint PrimitiveReference;
#endif
void main()
{
#if defined(EVANS_VISIBILITY_RESOLVE)
    // Separate resolve variants keep inactive SSBO reads out of ANGLE shaders.
#if defined(EVANS_MATERIAL_TILES)
    {
        const uint QuadVertices=6u;
        const uvec2 Corners[QuadVertices]=uvec2[QuadVertices](uvec2(0,0),uvec2(1,0),uvec2(0,1),uvec2(0,1),uvec2(1,0),uvec2(1,1));
        uint list=MaterialTileWord(MaterialTiles.x);
        uint tile=MaterialTileWord(MaterialTiles.w+list+uint(gl_InstanceID));
        vec2 pixel=min(vec2((uvec2(tile%MaterialTiles.y,tile/MaterialTiles.y)+Corners[gl_VertexID])*MaterialTiles.z),Extent.xy);
        gl_Position=vec4(pixel*Extent.zw*2.0-1.0,0.0,1.0); return;
    }
#else
    uvec2 corner = uvec2((uint(gl_VertexID) << 1u) & 2u, uint(gl_VertexID) & 2u);
    gl_Position = vec4(vec2(corner) * 2.0 - 1.0, 0.0, 1.0);
#endif
#else
    // Native lines use a separate variant: six corners rather than three.
    // This also avoids ANGLE miscompiling a material-dependent vertex-ID divisor.
    const uint EdgeVerticesPerTriangle = 6u;
#if defined(EVANS_VISIBILITY_WIREFRAME)
    const bool wire = true;
#else
    const bool wire = false;
#endif
    uint vertexCount = wire ? EdgeVerticesPerTriangle : VerticesPerTriangle;
    uint primitive = uint(gl_VertexID) / vertexCount, localVertex = uint(gl_VertexID) % vertexCount;
    uint corner = wire ? ((localVertex + 1u) / 2u) % VerticesPerTriangle : localVertex;
    uint index = DrawRange.y + primitive;
    Weights = vec2(0.0); UV = vec2(0.0); VertexOpacity = 0.0; PrimitiveReference = 0u; gl_Position = vec4(0.0);
    if (primitive >= DrawRange.z || index >= DrawRange.x) return;
    TriangleRecord triangle = ReadTriangle(index); gl_Position = ClipPosition(PositionAt(triangle, corner));
    if (wire && WireFaceSign!=0)
    {
        vec3 a=ClipPosition(triangle.Position0).xyw;
        vec3 b=ClipPosition(triangle.Position1).xyw;
        vec3 c=ClipPosition(triangle.Position2).xyw;
        if (dot(a,cross(b,c))*float(WireFaceSign)<=0.0)
        { gl_Position=vec4(0.0,0.0,2.0,1.0); return; }
    }
    gl_Position.z = gl_Position.z * 2.0 - gl_Position.w; // Renderer Z in [0,W] -> ES Z in [-W,W].
    Weights = vec2(corner == 1u ? 1.0 : 0.0, corner == 2u ? 1.0 : 0.0);
    UV = TextureCoordinate(triangle, corner); VertexOpacity = ColorAt(triangle, corner).a; PrimitiveReference = index + 1u;
#endif
}
#else
#if !defined(EVANS_VISIBILITY_RESOLVE)
in vec2 Weights;
in vec2 UV;
in float VertexOpacity;
flat in uint PrimitiveReference;
layout(location = 0) out highp uint StoredReference;
layout(location = 1) out vec2 StoredWeights;
void main()
{
    if (PrimitiveReference == 0u) discard;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (MaterialParameters.w > 1.5)
    {
        vec3 wireBary = vec3(1.0 - Weights.x - Weights.y, Weights);
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    float alpha = VertexOpacity * MaterialParameters.y;
    if (TextureFlags.x != 0u) alpha *= texture(AlbedoImage, UV).a;
    if (TextureFlags.z != 0u) alpha *= texture(OpacityImage, UV).r;
    if (MaterialParameters.z > 0.0 && clamp(alpha, 0.0, 1.0) < MaterialParameters.z) discard;
    StoredReference = PrimitiveReference | (gl_FrontFacing ? 0u : BackFacing); StoredWeights = Weights;
}
#else
layout(location = 0) out vec4 DiffuseOpacity;
layout(location = 1) out vec4 NormalRoughness;
layout(location = 2) out vec4 SpecularShininess;
layout(location = 3) out vec4 AmbientMetallic;
layout(location = 4) out vec4 Emission;
bool BarycentricDerivatives(TriangleRecord triangle, vec2 topLeftPixel, out vec3 dx, out vec3 dy)
{
    dx = dy = vec3(0.0);
    vec3 a = ClipPosition(triangle.Position0).xyw, b = ClipPosition(triangle.Position1).xyw;
    vec3 c = ClipPosition(triangle.Position2).xyw;
    vec3 bounds = max(abs(a), max(abs(b), abs(c))); float scale = max(bounds.x, max(bounds.y, bounds.z));
    if (scale == 0.0) return false;
    a /= scale; b /= scale; c /= scale;
    vec3 inverseA = cross(b, c), inverseB = cross(c, a), inverseC = cross(a, b);
    float determinant = dot(a, inverseA);
    if (abs(determinant) <= MatrixTolerance) return false;
    inverseA /= determinant; inverseB /= determinant; inverseC /= determinant;
    vec3 point = vec3(topLeftPixel * Extent.zw * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 1.0);
    vec3 h = vec3(dot(inverseA, point), dot(inverseB, point), dot(inverseC, point)); float sum = h.x + h.y + h.z;
    if (abs(sum) <= MatrixTolerance) return false;
    vec3 hx = vec3(inverseA.x, inverseB.x, inverseC.x) * (2.0 * Extent.z);
    vec3 hy = vec3(inverseA.y, inverseB.y, inverseC.y) * (-2.0 * Extent.w);
    dx = (hx - h / sum * (hx.x + hx.y + hx.z)) / sum;
    dy = (hy - h / sum * (hy.x + hy.y + hy.z)) / sum;
    if (!any(isnan(dx)) && !any(isinf(dx)) && !any(isnan(dy)) && !any(isinf(dy))) return true;
    dx = dy = vec3(0.0); return false;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy); uint encoded = texelFetch(VisibilityImage, pixel, 0).r;
    uint reference = encoded & TriangleReferenceMask;
    // Keep helper-lane SSBO reads in range independently of fragment discard.
    uint safeTriangle = min(reference > 0u ? reference - 1u : 0u, max(DrawRange.x, 1u) - 1u);
    if (reference == 0u || reference > DrawRange.x) discard;
    TriangleRecord triangle = ReadTriangle(safeTriangle);
#if defined(EVANS_MATERIAL_TILES)
    if (MaterialBins[triangle.MaterialSource.x]!=MaterialTiles.x) discard;
#else
    if (triangle.MaterialSource.x!=DrawRange.w) discard;
#endif
    vec2 stored = texelFetch(BarycentricImage, pixel, 0).xy; vec3 bary = vec3(1.0 - stored.x - stored.y, stored);
    vec3 dx, dy;
    // Texture fetch coordinates retain ES bottom-left storage. Only the analytic
    // derivative coordinate is converted to the renderer's top-left convention.
    bool gradients = BarycentricDerivatives(triangle, vec2(gl_FragCoord.x, Extent.y - gl_FragCoord.y), dx, dy);
    vec2 uv0 = TextureCoordinate(triangle, 0u), uv1 = TextureCoordinate(triangle, 1u), uv2 = TextureCoordinate(triangle, 2u);
    vec2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    // Derivatives sum to zero: edge form avoids cancellation against the world origin.
    vec2 ux = (uv1 - uv0) * dx.y + (uv2 - uv0) * dx.z, uy = (uv1 - uv0) * dy.y + (uv2 - uv0) * dy.z;
    vec4 color = triangle.Color0 * bary.x + triangle.Color1 * bary.y + triangle.Color2 * bary.z;
    vec4 albedo = TextureFlags.x != 0u ? textureGrad(AlbedoImage, uv, ux, -uy) : vec4(1.0);
    if (float(TextureFlags.w) == ToonShadingModel && TextureFlags.x != 0u)
        albedo.rgb = ToonTextureColor(AlbedoImage, uv, ux, -uy);
    vec3 surfaceMap = TextureFlags.z != 0u ? textureGrad(OpacityImage, uv, ux, -uy).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = SurfaceParameters.xy;
    if (TextureFlags.z > 1u) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    vec3 normal = NormalizeOr(triangle.Normal0.xyz * bary.x + triangle.Normal1.xyz * bary.y + triangle.Normal2.xyz * bary.z, vec3(0.0, 1.0, 0.0));
    if (TextureFlags.w == FlatShading)
    {
        vec3 face = NormalizeOr(cross(triangle.Position1.xyz - triangle.Position0.xyz, triangle.Position2.xyz - triangle.Position0.xyz), normal);
        normal = dot(face, normal) < 0.0 ? -face : face;
    }
    if ((encoded & BackFacing) != 0u) normal = -normal;
    float determinant = ux.x * uy.y - ux.y * uy.x;
    if (gradients && TextureFlags.y != 0u && TextureFlags.w != FlatShading && abs(determinant) > UVTolerance)
    {
        vec3 px = (triangle.Position1.xyz - triangle.Position0.xyz) * dx.y + (triangle.Position2.xyz - triangle.Position0.xyz) * dx.z;
        vec3 py = (triangle.Position1.xyz - triangle.Position0.xyz) * dy.y + (triangle.Position2.xyz - triangle.Position0.xyz) * dy.z;
        vec3 tangent = (px * uy.y - py * ux.y) / determinant;
        vec3 originalBitangent = (py * ux.x - px * uy.x) / determinant;
        vec3 helper = abs(normal.y) < ParallelTangentLimit ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = NormalizeOr(tangent - normal * dot(normal, tangent), NormalizeOr(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
        vec3 bitangent = NormalizeOr(cross(normal, tangent), vec3(0.0, 0.0, 1.0));
        if (dot(bitangent, originalBitangent) < 0.0) bitangent = -bitangent;
        vec3 mapped = textureGrad(NormalImage, uv, ux, -uy).xyz * 2.0 - 1.0;
        if (float(TextureFlags.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
        normal = NormalizeOr(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
    }
    vec3 surface = ToonAlbedo(color.rgb * max(Tint.rgb, vec3(0.0)) * max(albedo.rgb, vec3(0.0)), float(TextureFlags.w));
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    vec3 ambient = ToonAmbient(surface * DiffuseMaterial.rgb, surface * max(AmbientMaterial.rgb, vec3(0.0)) * max(AmbientIllumination.rgb, vec3(0.0)) * max(AmbientIllumination.w, 0.0), float(TextureFlags.w));
    if (TextureFlags.w == PbrShading) ambient *= 1.0 - metallic;
    DiffuseOpacity = vec4(clamp(surface * DiffuseMaterial.rgb, 0.0, MaximumRadiance), clamp(color.a * MaterialParameters.y * albedo.a * opacity, 0.0, 1.0));
    NormalRoughness = vec4(normal, clamp(roughnessMetallic.x, MinimumRoughness, 1.0));
    SpecularShininess = clamp(vec4(SpecularMaterial.rgb, max(MaterialParameters.x, 1.0)), 0.0, MaximumRadiance);
    AmbientMetallic = vec4(clamp(ambient, 0.0, MaximumRadiance), metallic);
    Emission = vec4(clamp(EmissionMaterial.rgb, 0.0, MaximumRadiance), 1.0);
    if (MaterialParameters.w > 0.0) { NormalRoughness.w = -1.0; Emission.rgb = WireColor(Tint.rgb); }
}
#endif
#endif
