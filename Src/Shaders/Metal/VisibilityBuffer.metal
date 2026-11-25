#include <metal_stdlib>
using namespace metal;

// Native MSL implementation; all four stages belong to this technique. The
// raster pass stores triangle IDs and perspective-correct B/C weights. Material
// texture footprints are reconstructed analytically without cross-ID quads.
// Burns and Hunt (2013): https://jcgt.org/published/0002/02/04/
// Preserve Toon material color/detail before Deferred applies the light ramp.
// The analytic texture footprint filters RGB while keeping center alpha.
constant float ToonShadingModel = 4.0;
constant float ToonAlbedoExponent = 0.8;
constant float ToonPaletteSteps = 16.0;
constant float ToonColorSaturation = 1.12;
constant float ToonNormalStrength = 0.12;
constant float3 ToonLuma = float3(0.2126, 0.7152, 0.0722);
constant float3 ToonSkyFill = float3(0.24, 0.29, 0.4);
float3 ToonAlbedo(float3 value, float model)
{
    if (model != ToonShadingModel) return value;
    float3 paint = pow(saturate(value), float3(ToonAlbedoExponent));
    paint = floor(paint * ToonPaletteSteps + 0.5) / ToonPaletteSteps;
    return saturate(mix(float3(dot(paint, ToonLuma)), paint, ToonColorSaturation));
}
float3 ToonAmbient(float3 albedo, float3 ambient, float model)
{ return model == ToonShadingModel ? max(ambient, albedo * ToonSkyFill) : ambient; }
// Smooth only the texture color. Explicit gradients and the center alpha
// preserve the original material's mip selection and silhouette coverage.
constant float ToonTextureRadius = 1.25;
constant int ToonTextureKernelRadius = 1;
constant int ToonTextureKernelWidth = ToonTextureKernelRadius * 2 + 1;
constant float ToonTextureSampleWeight = 1.0 / (ToonTextureKernelWidth * ToonTextureKernelWidth);
float3 ToonTextureColor(texture2d<float> image, sampler surfaceSampler,
    float2 uv, float2 uvDx, float2 uvDy)
{
    float3 color = float3(0.0);
    for (int row = -ToonTextureKernelRadius; row <= ToonTextureKernelRadius; ++row)
        for (int column = -ToonTextureKernelRadius; column <= ToonTextureKernelRadius; ++column)
            color += image.sample(surfaceSampler, uv + ToonTextureRadius * (column * uvDx + row * uvDy), gradient2d(uvDx, uvDy)).rgb;
    return color * ToonTextureSampleWeight;
}

constant uint VisibilityBackFace = 0x80000000u, VisibilityReferenceMask = 0x7fffffffu;
constant uint TriangleCornerCount = 3u, FlatShadingModel = 1u, PbrShadingModel = 3u;
constant float InverseThreshold = 1.0e-12f, NormalThreshold = 1.0e-12f, TangentThreshold = 1.0e-10f;
constant float MinimumSurfaceRoughness = 0.045f, MaximumHalfRadiance = 65504.0f;
constant float ParallelTangentThreshold = 0.999f;
struct VisibilityTriangle
{
    float4 Position[3], Normal[3], Color[3];
    float4 UVAB, UVC;
    uint4 Source;
};
struct VisibilityFrame
{
    float4 CameraRows[4];
    float4 Extent, Tint, Diffuse, Ambient, Specular, Emission;
    float4 Material, AmbientLighting, Surface;
    uint4 Images, Draw, MaterialTiles;
};
float4 Project(float4 point, constant VisibilityFrame& frame)
{
    return float4(dot(frame.CameraRows[0], point), dot(frame.CameraRows[1], point),
        dot(frame.CameraRows[2], point), dot(frame.CameraRows[3], point));
}
float2 UVAt(VisibilityTriangle triangle, uint corner)
{
    float2 authored = corner == 0u ? triangle.UVAB.xy : corner == 1u ? triangle.UVAB.zw : triangle.UVC.xy;
    return float2(authored.x, 1.0f - authored.y);
}
float3 UnitOr(float3 value, float3 fallback)
{ return dot(value, value) > NormalThreshold ? normalize(value) : fallback; }
float3 WireLinearColor(float3 value)
{
    const float Threshold = 0.04045f, LinearScale = 12.92f, PowerScale = 1.055f, Offset = 0.055f, Exponent = 2.4f;
    value = saturate(value); float3 power = pow((value + Offset) / PowerScale, float3(Exponent));
    return select(power, value / LinearScale, value <= Threshold);
}
struct VisibilityRasterVertex
{
    float4 Position [[position]];
    float2 Barycentric;
    float2 UV;
    float Opacity;
    uint Reference [[flat]];
};
vertex VisibilityRasterVertex EvansVisibilityRasterVertex(uint vertexID [[vertex_id]],
    constant VisibilityFrame& frame [[buffer(0)]], const device VisibilityTriangle* triangles [[buffer(1)]])
{
    VisibilityRasterVertex result{};
    uint primitive = vertexID / TriangleCornerCount, corner = vertexID % TriangleCornerCount;
    uint index = frame.Draw.y + primitive;
    if (primitive >= frame.Draw.z || index >= frame.Draw.x) return result;
    VisibilityTriangle triangle = triangles[index];
    result.Position = Project(triangle.Position[corner], frame);
    result.Barycentric = float2(corner == 1u ? 1.0f : 0.0f, corner == 2u ? 1.0f : 0.0f);
    result.UV = UVAt(triangle, corner); result.Opacity = triangle.Color[corner].a; result.Reference = index + 1u;
    return result;
}
struct VisibilityRasterOutput { uint Reference [[color(0)]]; float2 Weights [[color(1)]]; };
fragment VisibilityRasterOutput EvansVisibilityRasterFragment(VisibilityRasterVertex input [[stage_in]],
    bool frontFace [[front_facing]], constant VisibilityFrame& frame [[buffer(0)]],
    texture2d<float> diffuse [[texture(2)]], texture2d<float> opacity [[texture(4)]], sampler repeatSampler [[sampler(0)]])
{
    if (input.Reference == 0u) discard_fragment();
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (frame.Material.w > 1.5)
    {
        float3 wireBary = float3(1 - input.Barycentric.x - input.Barycentric.y, input.Barycentric);
        float3 wireDx = dfdx(wireBary), wireDy = dfdy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard_fragment();
    }
    float coverage = input.Opacity * frame.Material.y;
    if (frame.Images.x != 0u) coverage *= diffuse.sample(repeatSampler, input.UV).a;
    if (frame.Images.z != 0u) coverage *= opacity.sample(repeatSampler, input.UV).r;
    if (frame.Material.z > 0.0f && saturate(coverage) < frame.Material.z) discard_fragment();
    return {input.Reference | (frontFace ? 0u : VisibilityBackFace), input.Barycentric};
}
struct IndexedVisibilityFragment
{
    float4 Position [[position]];
    float3 PlaneX [[flat]],PlaneY [[flat]],PlaneOrigin [[flat]];
    float2 UV; float Opacity;
    uint Reference [[flat]];
};
fragment VisibilityRasterOutput EvansVisibilityMeshFragment(IndexedVisibilityFragment input [[stage_in]],
    bool frontFace [[front_facing]],constant VisibilityFrame& frame [[buffer(0)]],
    const device VisibilityTriangle* triangles [[buffer(1)]],texture2d<float> diffuse [[texture(2)]],
    texture2d<float> opacity [[texture(4)]],sampler repeatSampler [[sampler(0)]])
{
    if (input.Reference==0 || input.Reference>frame.Draw.x) discard_fragment();
    float2 screen=input.Position.xy*frame.Extent.zw*float2(2,-2)+float2(-1,1);
    float3 h=input.PlaneX*screen.x+input.PlaneY*screen.y+input.PlaneOrigin; float denominator=h.x+h.y+h.z;
    if (denominator==0 || !all(isfinite(h))) discard_fragment();
    float3 bary=h/denominator; if (!all(isfinite(bary))) discard_fragment();
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (frame.Material.w > 1.5)
    {
        float3 wireBary = bary;
        float3 wireDx = dfdx(wireBary), wireDy = dfdy(wireBary);
        float3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), 1.0e-8);
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard_fragment();
    }
    if (frame.Material.z>0)
    {
        float2 uv=input.UV; float coverage=input.Opacity*frame.Material.y;
        if (frame.Images.x!=0) coverage*=diffuse.sample(repeatSampler,uv).a;
        if (frame.Images.z!=0) coverage*=opacity.sample(repeatSampler,uv).r;
        if (saturate(coverage)<frame.Material.z) discard_fragment();
    }
    return {input.Reference|(frontFace?0u:VisibilityBackFace),bary.yz};
}
struct VisibilityScreenVertex { float4 Position [[position]]; };
vertex VisibilityScreenVertex EvansVisibilityResolveVertex(uint vertexID [[vertex_id]],uint instanceID [[instance_id]],
    constant VisibilityFrame& frame [[buffer(0)]],const device uint* work [[buffer(3)]])
{
    if (frame.MaterialTiles.w!=0u)
    {
        const uint HeaderWords=8u, ListWord=4u;
        const uint2 corners[6]={uint2(0,0),uint2(1,0),uint2(0,1),uint2(0,1),uint2(1,0),uint2(1,1)};
        uint4 tiles=frame.MaterialTiles;
        uint list=work[tiles.x*HeaderWords+ListWord], tile=work[list+instanceID];
        float2 pixel=min(float2((uint2(tile%tiles.y,tile/tiles.y)+corners[vertexID])*tiles.z),frame.Extent.xy);
        return {float4(pixel*frame.Extent.zw*float2(2,-2)+float2(-1,1),0,1)};
    }
    uint2 corner((vertexID << 1u) & 2u, vertexID & 2u);
    return {float4(float2(corner) * float2(2, -2) + float2(-1, 1), 0, 1)};
}
bool VisibilityDerivatives(VisibilityTriangle triangle, float2 pixel, constant VisibilityFrame& frame,
    thread float3& dx, thread float3& dy)
{
    dx = dy = float3(0);
    float3 a = Project(triangle.Position[0], frame).xyw, b = Project(triangle.Position[1], frame).xyw;
    float3 c = Project(triangle.Position[2], frame).xyw;
    float3 bound = max(abs(a), max(abs(b), abs(c))); float scale = max(bound.x, max(bound.y, bound.z));
    if (scale == 0.0f) return false;
    a /= scale; b /= scale; c /= scale;
    float3 row0 = cross(b, c), row1 = cross(c, a), row2 = cross(a, b);
    float determinant = dot(a, row0);
    if (abs(determinant) <= InverseThreshold) return false;
    row0 /= determinant; row1 /= determinant; row2 /= determinant;
    float3 point(pixel * frame.Extent.zw * float2(2, -2) + float2(-1, 1), 1);
    float3 h(dot(row0, point), dot(row1, point), dot(row2, point)); float denominator = h.x + h.y + h.z;
    if (abs(denominator) <= InverseThreshold) return false;
    float3 hx = float3(row0.x, row1.x, row2.x) * 2.0f * frame.Extent.z;
    float3 hy = float3(row0.y, row1.y, row2.y) * -2.0f * frame.Extent.w;
    dx = (hx - h / denominator * (hx.x + hx.y + hx.z)) / denominator;
    dy = (hy - h / denominator * (hy.x + hy.y + hy.z)) / denominator;
    if (all(isfinite(dx)) && all(isfinite(dy))) return true;
    dx = dy = float3(0); return false;
}
struct VisibilityMaterialOutput
{
    float4 Diffuse [[color(0)]]; float4 Normal [[color(1)]]; float4 Specular [[color(2)]];
    float4 Ambient [[color(3)]]; float4 Emission [[color(4)]];
};
fragment VisibilityMaterialOutput EvansVisibilityResolveFragment(VisibilityScreenVertex input [[stage_in]],
    const device uint* materialBins [[buffer(2)]],
    constant VisibilityFrame& frame [[buffer(0)]], const device VisibilityTriangle* triangles [[buffer(1)]],
    texture2d<uint, access::read> references [[texture(0)]], texture2d<float, access::read> weights [[texture(1)]],
    texture2d<float> diffuse [[texture(2)]], texture2d<float> normalMap [[texture(3)]],
    texture2d<float> opacityMap [[texture(4)]], sampler repeatSampler [[sampler(0)]])
{
    uint2 pixel = uint2(input.Position.xy); uint encoded = references.read(pixel).r, reference = encoded & VisibilityReferenceMask;
    // A discarded fragment/helper must still use a valid device-buffer address.
    uint safeTriangle = min(reference > 0u ? reference - 1u : 0u, max(frame.Draw.x, 1u) - 1u);
    if (reference == 0u || reference > frame.Draw.x) discard_fragment();
    VisibilityTriangle triangle = triangles[safeTriangle];
    uint owner=triangle.Source.x;
    if (frame.MaterialTiles.w!=0u) owner=materialBins[owner];
    if (owner!=(frame.MaterialTiles.w!=0u?frame.MaterialTiles.x:frame.Draw.w)) discard_fragment();
    float2 stored = weights.read(pixel).xy; float3 bary(1.0f - stored.x - stored.y, stored);
    float3 dx, dy; bool gradients = VisibilityDerivatives(triangle, input.Position.xy, frame, dx, dy);
    float2 a = UVAt(triangle, 0u), b = UVAt(triangle, 1u), c = UVAt(triangle, 2u);
    float2 uv = a * bary.x + b * bary.y + c * bary.z;
    // Derivatives sum to zero: edge form avoids cancellation against the world origin.
    float2 ux = (b - a) * dx.y + (c - a) * dx.z, uy = (b - a) * dy.y + (c - a) * dy.z;
    float4 color = triangle.Color[0] * bary.x + triangle.Color[1] * bary.y + triangle.Color[2] * bary.z;
    float4 albedo = frame.Images.x != 0u ? diffuse.sample(repeatSampler, uv, gradient2d(ux, uy)) : float4(1);
    if (float(frame.Images.w) == ToonShadingModel && frame.Images.x != 0u)
        albedo.rgb = ToonTextureColor(diffuse, repeatSampler, uv, ux, uy);
    float3 surfaceMap = frame.Images.z != 0 ? opacityMap.sample(repeatSampler, uv, gradient2d(ux, uy)).rgb : float3(1, 1, 1);
    float2 roughnessMetallic = frame.Surface.xy;
    if (frame.Images.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    float3 normal = UnitOr(triangle.Normal[0].xyz * bary.x + triangle.Normal[1].xyz * bary.y + triangle.Normal[2].xyz * bary.z, float3(0, 1, 0));
    if (frame.Images.w == FlatShadingModel)
    {
        float3 face = UnitOr(cross(triangle.Position[1].xyz - triangle.Position[0].xyz, triangle.Position[2].xyz - triangle.Position[0].xyz), normal);
        normal = dot(face, normal) < 0.0f ? -face : face;
    }
    if ((encoded & VisibilityBackFace) != 0u) normal = -normal;
    float determinant = ux.x * uy.y - ux.y * uy.x;
    if (gradients && frame.Images.y != 0u && frame.Images.w != FlatShadingModel && abs(determinant) > TangentThreshold)
    {
        float3 px = (triangle.Position[1].xyz - triangle.Position[0].xyz) * dx.y + (triangle.Position[2].xyz - triangle.Position[0].xyz) * dx.z;
        float3 py = (triangle.Position[1].xyz - triangle.Position[0].xyz) * dy.y + (triangle.Position[2].xyz - triangle.Position[0].xyz) * dy.z;
        float3 tangent = (px * uy.y - py * ux.y) / determinant;
        float3 sourceBitangent = (py * ux.x - px * uy.x) / determinant;
        float3 helper = abs(normal.y) < ParallelTangentThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        tangent = UnitOr(tangent - normal * dot(normal, tangent), UnitOr(cross(helper, normal), float3(1, 0, 0)));
        float3 bitangent = UnitOr(cross(normal, tangent), float3(0, 0, 1));
        if (dot(bitangent, sourceBitangent) < 0.0f) bitangent = -bitangent;
        float3 mapped = normalMap.sample(repeatSampler, uv, gradient2d(ux, uy)).xyz * 2.0f - 1.0f;
        if (float(frame.Images.w) == ToonShadingModel) mapped = mix(float3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
        normal = UnitOr(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
    }
    float3 surface = ToonAlbedo(color.rgb * max(frame.Tint.rgb, float3(0)) * max(albedo.rgb, float3(0)), float(frame.Images.w));
    float metallic = saturate(roughnessMetallic.y);
    float3 ambient = ToonAmbient(surface * frame.Diffuse.rgb, surface * max(frame.Ambient.rgb, float3(0)) * max(frame.AmbientLighting.rgb, float3(0)) * max(frame.AmbientLighting.w, 0.0f), float(frame.Images.w));
    if (frame.Images.w == PbrShadingModel) ambient *= 1.0f - metallic;
    VisibilityMaterialOutput result;
    result.Diffuse = float4(clamp(surface * frame.Diffuse.rgb, 0.0f, MaximumHalfRadiance), saturate(color.a * frame.Material.y * albedo.a * opacity));
    result.Normal = float4(normal, clamp(roughnessMetallic.x, MinimumSurfaceRoughness, 1.0f));
    result.Specular = clamp(float4(frame.Specular.rgb, max(frame.Material.x, 1.0f)), 0.0f, MaximumHalfRadiance);
    result.Ambient = float4(clamp(ambient, 0.0f, MaximumHalfRadiance), metallic);
    result.Emission = float4(clamp(frame.Emission.rgb, 0.0f, MaximumHalfRadiance), 1);
    if (frame.Material.w > 0.0f) { result.Normal.w = -1; result.Emission.rgb = WireLinearColor(frame.Tint.rgb); }
    return result;
}
