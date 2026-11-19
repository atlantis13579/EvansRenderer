// D3D12 owns this source and its byte-addressed triangle reader. Rasterization
// writes R32_UINT references and RG32_FLOAT perspective barycentrics; a material
// resolve emits the five explicit deferred attachments. No bindless textures.
// Burns and Hunt (2013): https://jcgt.org/published/0002/02/04/
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

static const uint BackFacing = 0x80000000u, ReferenceMask = 0x7fffffffu;
static const uint TriangleBytes = 192u, VectorBytes = 16u, CornerCount = 3u;
static const uint NormalOffset = 48u, ColorOffset = 96u, UVOffset = 144u, SourceOffset = 176u;
static const float MatrixEpsilon = 1.0e-12f, UnitEpsilon = 1.0e-12f, UVEpsilon = 1.0e-10f;
static const float HalfFloatMaximum = 65504.0f, RoughnessFloor = 0.045f;
static const float TangentParallelLimit = 0.999f;
static const uint FlatShading = 1u, PbrShading = 3u;
cbuffer VisibilityParameters : register(b0)
{
    row_major float4x4 Camera;
    float4 Extent, Tint, DiffuseReflectance, AmbientReflectance, SpecularReflectance, EmittedRadiance;
    float4 MaterialParameters, AmbientLight, SurfaceParameters;
    uint4 TextureOptions, DrawRange, MaterialTiles;
};
ByteAddressBuffer TriangleStorage : register(t0);
static const uint MaterialWordBytes=4u;
ByteAddressBuffer MaterialBins : register(t6);
ByteAddressBuffer MaterialWork : register(t7);
Texture2D<uint> References : register(t1);
Texture2D<float2> InterpolationWeights : register(t2);
Texture2D<float4> AlbedoTexture : register(t3);
Texture2D<float4> NormalTexture : register(t4);
Texture2D<float4> CoverageTexture : register(t5);
SamplerState MaterialSampler : register(s0);
struct TriangleData { float4 P[3], N[3], C[3]; float2 UV[3]; uint Material; };
TriangleData ReadTriangle(uint index)
{
    uint base = index * TriangleBytes;
    TriangleData result;
    [unroll] for (uint corner = 0u; corner < CornerCount; ++corner)
    {
        result.P[corner] = asfloat(TriangleStorage.Load4(base + corner * VectorBytes));
        result.N[corner] = asfloat(TriangleStorage.Load4(base + NormalOffset + corner * VectorBytes));
        result.C[corner] = asfloat(TriangleStorage.Load4(base + ColorOffset + corner * VectorBytes));
    }
    float4 uvAB = asfloat(TriangleStorage.Load4(base + UVOffset));
    float2 uvC = asfloat(TriangleStorage.Load2(base + UVOffset + VectorBytes));
    result.UV[0] = float2(uvAB.x, 1 - uvAB.y); result.UV[1] = float2(uvAB.z, 1 - uvAB.w);
    result.UV[2] = float2(uvC.x, 1 - uvC.y); result.Material = TriangleStorage.Load(base + SourceOffset);
    return result;
}
float3 SafeUnit(float3 value, float3 fallback)
{ return dot(value, value) > UnitEpsilon ? normalize(value) : fallback; }
float3 DecodeWireTint(float3 value)
{
    const float LinearThreshold = 0.04045f, LinearDivisor = 12.92f, PowerDivisor = 1.055f, PowerOffset = 0.055f, Exponent = 2.4f;
    value = saturate(value); float3 power = pow((value + PowerOffset) / PowerDivisor, Exponent);
    return float3(value.r <= LinearThreshold ? value.r / LinearDivisor : power.r,
        value.g <= LinearThreshold ? value.g / LinearDivisor : power.g, value.b <= LinearThreshold ? value.b / LinearDivisor : power.b);
}
struct VisibilityVertex
{
    float4 Position : SV_POSITION;
    float2 Weights : TEXCOORD0;
    float2 UV : TEXCOORD1;
    float Alpha : TEXCOORD2;
    nointerpolation uint ID : TEXCOORD3;
};
VisibilityVertex VSRaster(uint vertexID : SV_VertexID)
{
    VisibilityVertex result = (VisibilityVertex)0;
    uint primitive = vertexID / CornerCount, corner = vertexID % CornerCount;
    uint index = DrawRange.y + primitive;
    if (primitive >= DrawRange.z || index >= DrawRange.x) return result;
    TriangleData data = ReadTriangle(index);
    result.Position = mul(Camera, data.P[corner]); result.UV = data.UV[corner];
    result.Weights = float2(corner == 1u, corner == 2u); result.Alpha = data.C[corner].a; result.ID = index + 1u;
    return result;
}
struct VisibilityOutput { uint ID : SV_TARGET0; float2 Weights : SV_TARGET1; };
VisibilityOutput PSRaster(VisibilityVertex input, bool frontFace : SV_IsFrontFace)
{
    if (input.ID == 0u) discard;
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (MaterialParameters.w > 1.5)
    {
        float3 wireBary = float3(1 - input.Weights.x - input.Weights.y, input.Weights);
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    float coverage = input.Alpha * MaterialParameters.y;
    if (TextureOptions.x != 0u) coverage *= AlbedoTexture.Sample(MaterialSampler, input.UV).a;
    if (TextureOptions.z != 0u) coverage *= CoverageTexture.Sample(MaterialSampler, input.UV).r;
    if (MaterialParameters.z > 0 && saturate(coverage) < MaterialParameters.z) discard;
    VisibilityOutput result; result.ID = input.ID | (frontFace ? 0u : BackFacing); result.Weights = input.Weights; return result;
}
// Indexed mesh vertices share positions. Primitive-local interpolation planes
// retain attribute seams; cutout UV/alpha keep native raster interpolation.
struct MeshRasterFragment
{
    float4 Position:SV_POSITION;
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    nointerpolation float3 PlaneX:TEXCOORD0,PlaneY:TEXCOORD1,PlaneOrigin:TEXCOORD2;
    float2 UV:TEXCOORD4; float Alpha:TEXCOORD5;
#endif
    nointerpolation uint ID:TEXCOORD3;
};
VisibilityOutput PSMeshRaster(MeshRasterFragment input,bool frontFace:SV_IsFrontFace
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    ,float3 bary:SV_Barycentrics
#endif
)
{
    if (input.ID==0u || input.ID>DrawRange.x) discard;
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    float2 screen=input.Position.xy*Extent.zw*float2(2,-2)+float2(-1,1);
    precise float3 h=input.PlaneX*screen.x+input.PlaneY*screen.y+input.PlaneOrigin;
    float denominator=h.x+h.y+h.z;
    if (denominator==0 || !all(isfinite(h))) discard;
    float3 bary=h/denominator;
#endif
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (MaterialParameters.w > 1.5)
    {
        float3 wireBary = bary;
        float3 wireDx = ddx(wireBary), wireDy = ddy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    if (!all(isfinite(bary))) discard;
    if (MaterialParameters.z>0)
    {
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        TriangleData data=ReadTriangle(min(input.ID-1u,max(DrawRange.x,1u)-1u));
        float2 uv=data.UV[0]*bary.x+data.UV[1]*bary.y+data.UV[2]*bary.z;
        float coverage=dot(float3(data.C[0].a,data.C[1].a,data.C[2].a),bary)*MaterialParameters.y;
#else
        float2 uv=input.UV; float coverage=input.Alpha*MaterialParameters.y;
#endif
        if (TextureOptions.x!=0u) coverage*=AlbedoTexture.Sample(MaterialSampler,uv).a;
        if (TextureOptions.z!=0u) coverage*=CoverageTexture.Sample(MaterialSampler,uv).r;
        if (saturate(coverage)<MaterialParameters.z) discard;
    }
    VisibilityOutput result; result.ID=input.ID|(frontFace?0u:BackFacing); result.Weights=bary.yz; return result;
}
struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSResolve(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    [branch] if (MaterialTiles.w!=0u)
    {
        const uint HeaderWords=8u, ListOffset=4u, WordBytes=4u;
        const uint2 Corners[6]={uint2(0,0),uint2(1,0),uint2(0,1),uint2(0,1),uint2(1,0),uint2(1,1)};
        uint list=MaterialWork.Load((MaterialTiles.x*HeaderWords+ListOffset)*WordBytes);
        uint tile=MaterialWork.Load((list+instanceID)*WordBytes);
        uint2 position=(uint2(tile%MaterialTiles.y,tile/MaterialTiles.y)+Corners[vertexID])*MaterialTiles.z;
        float2 pixel=min(float2(position),Extent.xy);
        ScreenVertex result; result.Position=float4(pixel*Extent.zw*float2(2,-2)+float2(-1,1),0,1); return result;
    }
    float2 corner = float2((vertexID << 1u) & 2u, vertexID & 2u);
    ScreenVertex result; result.Position = float4(corner * float2(2, -2) + float2(-1, 1), 0, 1); return result;
}
bool AttributeDerivatives(TriangleData data, float2 pixel, out float3 ddxBary, out float3 ddyBary)
{
    ddxBary = ddyBary = 0;
    float3 a = mul(Camera, data.P[0]).xyw, b = mul(Camera, data.P[1]).xyw, c = mul(Camera, data.P[2]).xyw;
    float3 bounds = max(abs(a), max(abs(b), abs(c))); float scale = max(bounds.x, max(bounds.y, bounds.z));
    if (scale == 0) return false;
    a /= scale; b /= scale; c /= scale;
    float3 rowA = cross(b, c), rowB = cross(c, a), rowC = cross(a, b);
    float determinant = dot(a, rowA);
    if (abs(determinant) <= MatrixEpsilon) return false;
    rowA /= determinant; rowB /= determinant; rowC /= determinant;
    float3 screen = float3(pixel * Extent.zw * float2(2, -2) + float2(-1, 1), 1);
    float3 homogeneous = float3(dot(rowA, screen), dot(rowB, screen), dot(rowC, screen));
    float denominator = homogeneous.x + homogeneous.y + homogeneous.z;
    if (abs(denominator) <= MatrixEpsilon) return false;
    float3 derivativeX = float3(rowA.x, rowB.x, rowC.x) * 2 * Extent.z;
    float3 derivativeY = float3(rowA.y, rowB.y, rowC.y) * -2 * Extent.w;
    float3 bary = homogeneous / denominator;
    ddxBary = (derivativeX - bary * (derivativeX.x + derivativeX.y + derivativeX.z)) / denominator;
    ddyBary = (derivativeY - bary * (derivativeY.x + derivativeY.y + derivativeY.z)) / denominator;
    if (all(isfinite(ddxBary)) && all(isfinite(ddyBary))) return true;
    ddxBary = ddyBary = 0; return false;
}
struct MaterialOutput
{
    float4 Diffuse : SV_TARGET0; float4 Normal : SV_TARGET1; float4 Specular : SV_TARGET2;
    float4 Ambient : SV_TARGET3; float4 Emission : SV_TARGET4;
};
MaterialOutput PSResolve(ScreenVertex input)
{
    int2 pixel = int2(input.Position.xy); uint encoded = References.Load(int3(pixel, 0));
    uint reference = encoded & ReferenceMask;
    // Discarded helper lanes can still execute buffer loads. A root SRV has no
    // hardware range descriptor, so even background references need a safe index.
    uint safeTriangle = min(reference > 0u ? reference - 1u : 0u, max(DrawRange.x, 1u) - 1u);
    if (reference == 0u || reference > DrawRange.x) discard;
    TriangleData data = ReadTriangle(safeTriangle);
    uint owner=data.Material;
    [branch] if (MaterialTiles.w!=0u) owner=MaterialBins.Load(owner*MaterialWordBytes);
    if (owner != (MaterialTiles.w!=0u?MaterialTiles.x:DrawRange.w)) discard;
    float2 stored = InterpolationWeights.Load(int3(pixel, 0)); float3 bary = float3(1 - stored.x - stored.y, stored);
    float3 bx, by; bool hasDerivatives = AttributeDerivatives(data, input.Position.xy, bx, by);
    float2 uv = data.UV[0] * bary.x + data.UV[1] * bary.y + data.UV[2] * bary.z;
    // Derivatives sum to zero: edge form avoids cancellation against the world origin.
    float2 ux = (data.UV[1] - data.UV[0]) * bx.y + (data.UV[2] - data.UV[0]) * bx.z;
    float2 uy = (data.UV[1] - data.UV[0]) * by.y + (data.UV[2] - data.UV[0]) * by.z;
    float4 color = data.C[0] * bary.x + data.C[1] * bary.y + data.C[2] * bary.z;
    float4 albedo = TextureOptions.x != 0u ? AlbedoTexture.SampleGrad(MaterialSampler, uv, ux, uy) : 1;
    if (float(TextureOptions.w) == ToonShadingModel && TextureOptions.x != 0u)
        albedo.rgb = ToonTextureColor(AlbedoTexture, MaterialSampler, uv, ux, uy);
    float3 surfaceMap = TextureOptions.z != 0 ? CoverageTexture.SampleGrad(MaterialSampler, uv, ux, uy).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = SurfaceParameters.xy;
    if (TextureOptions.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float3 normal = SafeUnit(data.N[0].xyz * bary.x + data.N[1].xyz * bary.y + data.N[2].xyz * bary.z, float3(0, 1, 0));
    if (TextureOptions.w == FlatShading)
    {
        float3 geometric = SafeUnit(cross(data.P[1].xyz - data.P[0].xyz, data.P[2].xyz - data.P[0].xyz), normal);
        normal = dot(geometric, normal) < 0 ? -geometric : geometric;
    }
    if ((encoded & BackFacing) != 0u) normal = -normal;
    float uvDeterminant = ux.x * uy.y - ux.y * uy.x;
    if (hasDerivatives && TextureOptions.y != 0u && TextureOptions.w != FlatShading && abs(uvDeterminant) > UVEpsilon)
    {
        float3 px = (data.P[1].xyz - data.P[0].xyz) * bx.y + (data.P[2].xyz - data.P[0].xyz) * bx.z;
        float3 py = (data.P[1].xyz - data.P[0].xyz) * by.y + (data.P[2].xyz - data.P[0].xyz) * by.z;
        float3 tangent = (px * uy.y - py * ux.y) / uvDeterminant;
        float3 sourceBitangent = (py * ux.x - px * uy.x) / uvDeterminant;
        float3 helper = abs(normal.y) < TangentParallelLimit ? float3(0, 1, 0) : float3(1, 0, 0);
        tangent = SafeUnit(tangent - normal * dot(tangent, normal), SafeUnit(cross(helper, normal), float3(1, 0, 0)));
        float3 bitangent = SafeUnit(cross(normal, tangent), float3(0, 0, 1));
        if (dot(bitangent, sourceBitangent) < 0) bitangent = -bitangent;
        float3 mapped = NormalTexture.SampleGrad(MaterialSampler, uv, ux, uy).xyz * 2 - 1;
        if (float(TextureOptions.w) == ToonShadingModel) mapped = lerp(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
        normal = SafeUnit(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
    }
    float3 surface = ToonAlbedo(color.rgb * max(Tint.rgb, 0) * max(albedo.rgb, 0), float(TextureOptions.w));
    float metallic = saturate(roughnessMetallic.y);
    float3 ambient = ToonAmbient(surface * DiffuseReflectance.rgb, surface * max(AmbientReflectance.rgb, 0) * max(AmbientLight.rgb, 0) * max(AmbientLight.w, 0), float(TextureOptions.w));
    if (TextureOptions.w == PbrShading) ambient *= 1 - metallic;
    MaterialOutput result;
    result.Diffuse = float4(clamp(surface * DiffuseReflectance.rgb, 0, HalfFloatMaximum), saturate(color.a * MaterialParameters.y * albedo.a * opacity));
    result.Normal = float4(normal, clamp(roughnessMetallic.x, RoughnessFloor, 1));
    result.Specular = clamp(float4(SpecularReflectance.rgb, max(MaterialParameters.x, 1)), 0, HalfFloatMaximum);
    result.Ambient = float4(clamp(ambient, 0, HalfFloatMaximum), metallic);
    result.Emission = float4(clamp(EmittedRadiance.rgb, 0, HalfFloatMaximum), 1);
    if (MaterialParameters.w > 0) { result.Normal.w = -1; result.Emission.rgb = DecodeWireTint(Tint.rgb); }
    return result;
}
