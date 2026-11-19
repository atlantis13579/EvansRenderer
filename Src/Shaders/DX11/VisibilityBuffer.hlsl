// D3D11 visibility rasterization and deferred material reconstruction.
// Burns and Hunt, JCGT 2(2), 2013: https://jcgt.org/published/0002/02/04/
// Integer references use nearest reads. Barycentrics are perspective-correct
// interpolants; analytic texture gradients never differentiate across IDs.
// Preserve Toon material color/detail before Deferred applies the light ramp.
// The analytic texture footprint filters RGB while keeping center alpha.
static const float ToonShadingModel = 4.0;
static const float ToonAlbedoExponent = 0.8;
static const float ToonPaletteSteps = 16.0;
static const float ToonColorSaturation = 1.12;
static const float ToonNormalStrength = 0.12;
static const float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
static const float3 ToonSkyFill = float3(0.24, 0.29, 0.4);
float3 ToonAlbedo(float3 color, float model)
{
    if (model != ToonShadingModel) return color;
    float3 paint = pow(saturate(color), ToonAlbedoExponent);
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(lerp(dot(paint, ToonLuma).xxx, paint, ToonColorSaturation));
}
float3 ToonAmbient(float3 albedo, float3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
// Integrate a small screen-space footprint in the existing diffuse texture.
// Alpha remains the center sample, preserving foliage and curtain coverage.
static const float ToonTextureRadius = 1.25;
static const int ToonTextureKernelRadius = 1;
static const int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
static const float ToonTextureSampleWeight = 1.0 / (ToonTextureKernelWidth * ToonTextureKernelWidth);
float3 ToonTextureColor(Texture2D<float4> image, SamplerState surfaceSampler,
    float2 uv, float2 uvDx, float2 uvDy)
{
    float3 color = 0;
    [unroll] for (int y = -ToonTextureKernelRadius; y <= ToonTextureKernelRadius; ++y)
        [unroll] for (int x = -ToonTextureKernelRadius; x <= ToonTextureKernelRadius; ++x)
            color += image.SampleGrad(surfaceSampler, uv + ToonTextureRadius * (x * uvDx + y * uvDy), uvDx, uvDy).rgb;
    return color * ToonTextureSampleWeight;
}

static const uint BackFaceBit = 0x80000000u, TriangleMask = 0x7fffffffu;
static const float InverseTolerance = 1.0e-12f, NormalTolerance = 1.0e-12f;
static const float TangentTolerance = 1.0e-10f, MinimumRoughness = 0.045f;
static const float MaximumRadiance = 65504.0f;
static const float ParallelAxisLimit = 0.999f;
static const uint FlatModel = 1u, PbrModel = 3u, TriangleCorners = 3u;
struct Triangle
{
    float4 Position[3], Normal[3], Color[3];
    float4 UVAB, UVC;
    uint4 Source;
};
cbuffer VisibilityFrame : register(b0)
{
    row_major float4x4 ViewProjection;
    float4 Viewport, PipelineColor, Diffuse, Ambient, Specular, Emission;
    float4 Material, AmbientLighting, Shading;
    uint4 Options, Draw, MaterialTiles;
};
StructuredBuffer<Triangle> Triangles : register(t0);
static const uint MaterialWordBytes=4u;
ByteAddressBuffer MaterialBins : register(t6);
ByteAddressBuffer MaterialWork : register(t7);
Texture2D<uint> Visibility : register(t1);
Texture2D<float2> Barycentrics : register(t2);
Texture2D<float4> DiffuseMap : register(t3);
Texture2D<float4> NormalMap : register(t4);
Texture2D<float4> OpacityMap : register(t5);
SamplerState SurfaceSampler : register(s0); // Repeat; diffuse view is sRGB, other views are linear.
float2 TextureUV(Triangle primitive, uint corner)
{
    float2 uv = corner == 0u ? primitive.UVAB.xy : corner == 1u ? primitive.UVAB.zw : primitive.UVC.xy;
    return float2(uv.x, 1 - uv.y);
}
float3 Unit(float3 value, float3 fallback)
{ return dot(value, value) > NormalTolerance ? normalize(value) : fallback; }
float3 DecodeWireColor(float3 value)
{
    const float Threshold = 0.04045f, LinearScale = 12.92f, Scale = 1.055f, Offset = 0.055f, Gamma = 2.4f;
    value = saturate(value); float3 power = pow((value + Offset) / Scale, Gamma);
    return float3(value.r <= Threshold ? value.r / LinearScale : power.r,
        value.g <= Threshold ? value.g / LinearScale : power.g, value.b <= Threshold ? value.b / LinearScale : power.b);
}
struct RasterVertex
{
    float4 Position : SV_POSITION;
    float2 Bary : TEXCOORD0;
    float2 UV : TEXCOORD1;
    float Opacity : TEXCOORD2;
    nointerpolation uint Reference : TEXCOORD3;
};
RasterVertex VSRaster(uint vertex : SV_VertexID)
{
    RasterVertex result = (RasterVertex)0;
    uint localTriangle = vertex / TriangleCorners, corner = vertex % TriangleCorners;
    uint index = Draw.y + localTriangle;
    if (localTriangle >= Draw.z || index >= Draw.x) return result;
    Triangle primitive = Triangles[index];
    result.Position = mul(ViewProjection, primitive.Position[corner]);
    result.Bary = float2(corner == 1u, corner == 2u);
    result.UV = TextureUV(primitive, corner); result.Opacity = primitive.Color[corner].a;
    result.Reference = index + 1u;
    return result;
}
struct VisibilityTargets { uint Reference : SV_TARGET0; float2 Bary : SV_TARGET1; };
VisibilityTargets PSRaster(RasterVertex input, bool front : SV_IsFrontFace)
{
    if (input.Reference == 0u) discard;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (Material.w > 1.5)
    {
        float3 wireBary = float3(1 - input.Bary.x - input.Bary.y, input.Bary);
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    float alpha = input.Opacity * Material.y;
    if (Options.x != 0u) alpha *= DiffuseMap.Sample(SurfaceSampler, input.UV).a;
    if (Options.z != 0u) alpha *= OpacityMap.Sample(SurfaceSampler, input.UV).r;
    if (Material.z > 0 && saturate(alpha) < Material.z) discard;
    VisibilityTargets result;
    result.Reference = input.Reference | (front ? 0u : BackFaceBit); result.Bary = input.Bary;
    return result;
}
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSResolve(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    [branch] if (MaterialTiles.w != 0u)
    {
        const uint HeaderWords=8u, ListWord=4u, WordBytes=4u;
        const uint2 Corners[6]={uint2(0,0),uint2(1,0),uint2(0,1),uint2(0,1),uint2(1,0),uint2(1,1)};
        uint first=MaterialWork.Load((MaterialTiles.x*HeaderWords+ListWord)*WordBytes);
        uint tile=MaterialWork.Load((first+instance)*WordBytes);
        float2 pixel=min(float2((uint2(tile%MaterialTiles.y,tile/MaterialTiles.y)+Corners[vertex])*MaterialTiles.z),Viewport.xy);
        ScreenVertex result; result.Position=float4(pixel*Viewport.zw*float2(2,-2)+float2(-1,1),0,1); return result;
    }
    float2 corner = float2((vertex << 1u) & 2u, vertex & 2u);
    ScreenVertex result; result.Position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1); return result;
}
bool BarycentricGradients(Triangle primitive, float2 pixel, out float3 dx, out float3 dy)
{
    // Invert [clipA.xyw clipB.xyw clipC.xyw], then differentiate h/sum(h).
    // No division by an original W: triangles intersecting W=0 remain valid.
    float4 a = mul(ViewProjection, primitive.Position[0]), b = mul(ViewProjection, primitive.Position[1]);
    float4 c = mul(ViewProjection, primitive.Position[2]);
    float3 bound = max(abs(a.xyw), max(abs(b.xyw), abs(c.xyw)));
    float scale = max(bound.x, max(bound.y, bound.z)); dx = dy = 0;
    if (scale == 0) return false;
    float3 ca = a.xyw / scale, cb = b.xyw / scale, cc = c.xyw / scale;
    float3 r0 = cross(cb, cc), r1 = cross(cc, ca), r2 = cross(ca, cb);
    float determinant = dot(ca, r0);
    if (abs(determinant) <= InverseTolerance) return false;
    r0 /= determinant; r1 /= determinant; r2 /= determinant;
    float3 samplePosition = float3(pixel * Viewport.zw * float2(2, -2) + float2(-1, 1), 1);
    float3 h = float3(dot(r0, samplePosition), dot(r1, samplePosition), dot(r2, samplePosition));
    float sum = h.x + h.y + h.z;
    if (abs(sum) <= InverseTolerance) return false;
    float3 hx = float3(r0.x, r1.x, r2.x) * (2 * Viewport.z);
    float3 hy = float3(r0.y, r1.y, r2.y) * (-2 * Viewport.w);
    dx = (hx - h / sum * (hx.x + hx.y + hx.z)) / sum;
    dy = (hy - h / sum * (hy.x + hy.y + hy.z)) / sum;
    if (all(isfinite(dx)) && all(isfinite(dy))) return true;
    dx = dy = 0; return false;
}
struct MaterialTargets
{
    float4 DiffuseOpacity : SV_TARGET0;
    float4 NormalRoughness : SV_TARGET1;
    float4 SpecularShininess : SV_TARGET2;
    float4 AmbientMetallic : SV_TARGET3;
    float4 EmissionColor : SV_TARGET4;
};
MaterialTargets PSResolve(ScreenVertex input)
{
    int2 pixel = int2(input.Position.xy);
    uint encoded = Visibility.Load(int3(pixel, 0)), reference = encoded & TriangleMask;
    // Helper lanes remain memory-safe even when discard does not end execution.
    uint safeTriangle = min(reference > 0u ? reference - 1u : 0u, max(Draw.x, 1u) - 1u);
    if (reference == 0u || reference > Draw.x) discard;
    Triangle primitive = Triangles[safeTriangle];
    uint owner=primitive.Source.x;
    [branch] if (MaterialTiles.w!=0u) owner=MaterialBins.Load(owner*MaterialWordBytes);
    if (owner != (MaterialTiles.w!=0u?MaterialTiles.x:Draw.w)) discard;
    float2 stored = Barycentrics.Load(int3(pixel, 0));
    float3 weights = float3(1 - stored.x - stored.y, stored);
    float3 dx, dy;
    bool gradients = BarycentricGradients(primitive, input.Position.xy, dx, dy);
    float2 uvA = TextureUV(primitive, 0u), uvB = TextureUV(primitive, 1u), uvC = TextureUV(primitive, 2u);
    float2 uv = uvA * weights.x + uvB * weights.y + uvC * weights.z;
    // Derivatives sum to zero: edge form avoids cancellation against the world origin.
    float2 uvDx = (uvB - uvA) * dx.y + (uvC - uvA) * dx.z;
    float2 uvDy = (uvB - uvA) * dy.y + (uvC - uvA) * dy.z;
    float4 color = primitive.Color[0] * weights.x + primitive.Color[1] * weights.y + primitive.Color[2] * weights.z;
    float4 texel = Options.x != 0u ? DiffuseMap.SampleGrad(SurfaceSampler, uv, uvDx, uvDy) : 1;
    if (float(Options.w) == ToonShadingModel && Options.x != 0u)
        texel.rgb = ToonTextureColor(DiffuseMap, SurfaceSampler, uv, uvDx, uvDy);
    float3 surfaceMap = Options.z != 0 ? OpacityMap.SampleGrad(SurfaceSampler, uv, uvDx, uvDy).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = Shading.xy;
    if (Options.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float3 normal = Unit(primitive.Normal[0].xyz * weights.x + primitive.Normal[1].xyz * weights.y + primitive.Normal[2].xyz * weights.z, float3(0, 1, 0));
    if (Options.w == FlatModel)
    {
        float3 face = Unit(cross(primitive.Position[1].xyz - primitive.Position[0].xyz,
            primitive.Position[2].xyz - primitive.Position[0].xyz), normal);
        normal = dot(face, normal) < 0 ? -face : face;
    }
    if ((encoded & BackFaceBit) != 0u) normal = -normal;
    float tangentDeterminant = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    if (gradients && Options.y != 0u && Options.w != FlatModel && abs(tangentDeterminant) > TangentTolerance)
    {
        float3 worldDx = (primitive.Position[1].xyz - primitive.Position[0].xyz) * dx.y + (primitive.Position[2].xyz - primitive.Position[0].xyz) * dx.z;
        float3 worldDy = (primitive.Position[1].xyz - primitive.Position[0].xyz) * dy.y + (primitive.Position[2].xyz - primitive.Position[0].xyz) * dy.z;
        float3 tangent = (worldDx * uvDy.y - worldDy * uvDx.y) / tangentDeterminant;
        float3 sourceB = (worldDy * uvDx.x - worldDx * uvDy.x) / tangentDeterminant;
        float3 helper = abs(normal.y) < ParallelAxisLimit ? float3(0, 1, 0) : float3(1, 0, 0);
        tangent = Unit(tangent - normal * dot(normal, tangent), Unit(cross(helper, normal), float3(1, 0, 0)));
        float3 bitangent = Unit(cross(normal, tangent), float3(0, 0, 1));
        if (dot(bitangent, sourceB) < 0) bitangent = -bitangent;
        float3 mapped = NormalMap.SampleGrad(SurfaceSampler, uv, uvDx, uvDy).xyz * 2 - 1;
        if (float(Options.w) == ToonShadingModel) mapped = lerp(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
        normal = Unit(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
    }
    float3 surface = ToonAlbedo(color.rgb * max(PipelineColor.rgb, 0) * max(texel.rgb, 0), float(Options.w));
    float metallic = saturate(roughnessMetallic.y);
    float3 ambient = ToonAmbient(surface * Diffuse.rgb, surface * max(Ambient.rgb, 0) * max(AmbientLighting.rgb, 0) * max(AmbientLighting.w, 0), float(Options.w));
    if (Options.w == PbrModel) ambient *= 1 - metallic;
    MaterialTargets result;
    result.DiffuseOpacity = float4(clamp(surface * Diffuse.rgb, 0, MaximumRadiance), saturate(color.a * Material.y * texel.a * opacity));
    result.NormalRoughness = float4(normal, clamp(roughnessMetallic.x, MinimumRoughness, 1));
    result.SpecularShininess = clamp(float4(Specular.rgb, max(Material.x, 1)), 0, MaximumRadiance);
    result.AmbientMetallic = float4(clamp(ambient, 0, MaximumRadiance), metallic);
    result.EmissionColor = float4(clamp(Emission.rgb, 0, MaximumRadiance), 1);
    if (Material.w > 0) { result.NormalRoughness.w = -1; result.EmissionColor.rgb = DecodeWireColor(PipelineColor.rgb); }
    return result;
}
