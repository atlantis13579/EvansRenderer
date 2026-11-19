#version 450
#if defined(EVANS_VISIBILITY_MESH)
#extension GL_EXT_mesh_shader : require
#endif
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
#extension GL_EXT_fragment_shader_barycentric : require
#endif
// Vulkan-native visibility stages, compiled from GLSL at runtime. Define
// EVANS_VERTEX_STAGE for a vertex stage and EVANS_VISIBILITY_RESOLVE for the
// material pass. R32_UINT/RG32_FLOAT attachments retain IDs and smooth barycentrics.
// Burns and Hunt (2013): https://jcgt.org/published/0002/02/04/
// Attribute gradients invert homogeneous XYW rather than divide original W.
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

const uint BackFace = 0x80000000u, ReferenceMask = 0x7fffffffu, CornerCount = 3u;
const uint FlatModel = 1u, PbrModel = 3u;
const float InverseEpsilon = 1.0e-12, NormalEpsilon = 1.0e-12, TangentEpsilon = 1.0e-10;
const float RoughnessMinimum = 0.045, HalfFloatMaximum = 65504.0;
const float ParallelAxisThreshold = 0.999;
struct Triangle
{
    vec4 Position[3]; vec4 Normal[3]; vec4 Color[3];
    vec4 UVAB; vec4 UVC; uvec4 Source;
};
layout(set = 0, binding = 0, std140, row_major) uniform Frame
{
    mat4 Camera;
    vec4 Viewport, Tint, Diffuse, Ambient, Specular, Emission;
    vec4 Material, AmbientLight, Surface;
    uvec4 Options, Draw, MaterialTiles;
} Parameters;
layout(set = 0, binding = 1, std430) readonly buffer TriangleTable { Triangle Records[]; } Geometry;
layout(set = 0, binding = 2) uniform usampler2D References;
layout(set = 0, binding = 3) uniform sampler2D Barycentrics;
layout(set = 0, binding = 4) uniform sampler2D DiffuseMap;
layout(set = 0, binding = 5) uniform sampler2D NormalMap;
layout(set = 0, binding = 6) uniform sampler2D OpacityMap;
layout(set=0,binding=7,std430) readonly buffer MaterialMapping { uint MaterialBins[]; };
layout(set=0,binding=8,std430) readonly buffer MaterialScheduling { uint MaterialWork[]; };
vec2 CornerUV(Triangle triangle, uint corner)
{
    vec2 uv = corner == 0u ? triangle.UVAB.xy : corner == 1u ? triangle.UVAB.zw : triangle.UVC.xy;
    return vec2(uv.x, 1.0 - uv.y);
}
vec3 Unit(vec3 value, vec3 fallback)
{ return dot(value, value) > NormalEpsilon ? normalize(value) : fallback; }
vec3 DecodeWire(vec3 value)
{
    const float Threshold = 0.04045, Divisor = 12.92, Scale = 1.055, Offset = 0.055, Exponent = 2.4;
    value = clamp(value, 0.0, 1.0); vec3 power = pow((value + Offset) / Scale, vec3(Exponent));
    return mix(power, value / Divisor, lessThanEqual(value, vec3(Threshold)));
}

#if defined(EVANS_VERTEX_STAGE)
invariant gl_Position;
precise gl_Position;
#if !defined(EVANS_VISIBILITY_RESOLVE)
layout(location = 0) out vec2 RasterBary;
layout(location = 1) out vec2 RasterUV;
layout(location = 2) out float RasterOpacity;
layout(location = 3) flat out uint RasterReference;
#endif
void main()
{
#if defined(EVANS_VISIBILITY_RESOLVE)
    if (Parameters.MaterialTiles.w!=0u)
    {
        const uint HeaderWords=8u, ListWord=4u;
        const uvec2 Corners[6]=uvec2[6](uvec2(0,0),uvec2(1,0),uvec2(0,1),uvec2(0,1),uvec2(1,0),uvec2(1,1));
        uvec4 tiles=Parameters.MaterialTiles;
        uint list=MaterialWork[tiles.x*HeaderWords+ListWord], tile=MaterialWork[list+uint(gl_InstanceIndex)];
        vec2 pixel=min(vec2((uvec2(tile%tiles.y,tile/tiles.y)+Corners[gl_VertexIndex])*tiles.z),Parameters.Viewport.xy);
        gl_Position=vec4(pixel*Parameters.Viewport.zw*2.0-1.0,0.0,1.0); return;
    }
    uvec2 corner = uvec2((uint(gl_VertexIndex) << 1u) & 2u, uint(gl_VertexIndex) & 2u);
    gl_Position = vec4(vec2(corner) * 2.0 - 1.0, 0.0, 1.0);
#else
    uint primitive = uint(gl_VertexIndex) / CornerCount, corner = uint(gl_VertexIndex) % CornerCount;
    uint index = Parameters.Draw.y + primitive;
    RasterBary = vec2(0.0); RasterUV = vec2(0.0); RasterOpacity = 0.0; RasterReference = 0u;
    gl_Position = vec4(0.0);
    if (primitive >= Parameters.Draw.z || index >= Parameters.Draw.x) return;
    Triangle triangle = Geometry.Records[index];
    gl_Position = Parameters.Camera * triangle.Position[corner];
    gl_Position.y = -gl_Position.y; // Positive-height Vulkan viewport; camera data stays renderer-space.
    RasterBary = vec2(corner == 1u ? 1.0 : 0.0, corner == 2u ? 1.0 : 0.0);
    RasterUV = CornerUV(triangle, corner); RasterOpacity = triangle.Color[corner].a; RasterReference = index + 1u;
#endif
}
#else
#if !defined(EVANS_VISIBILITY_RESOLVE)
#if defined(EVANS_VISIBILITY_MESH)
#if !defined(EVANS_MESH_NATIVE_BARYCENTRICS)
layout(location=0) perprimitiveEXT flat in vec3 MeshPlaneX;
layout(location=1) perprimitiveEXT flat in vec3 MeshPlaneY;
layout(location=2) perprimitiveEXT flat in vec3 MeshPlaneOrigin;
layout(location=4) in vec2 MeshCoverageUV;
layout(location=5) in float MeshCoverageAlpha;
#endif
layout(location=3) perprimitiveEXT flat in uint RasterReference;
#else
layout(location = 0) in vec2 RasterBary;
layout(location = 1) in vec2 RasterUV;
layout(location = 2) in float RasterOpacity;
layout(location = 3) flat in uint RasterReference;
#endif
layout(location = 0) out uint VisibilityID;
layout(location = 1) out vec2 VisibilityBary;
void main()
{
    if (RasterReference == 0u || RasterReference>Parameters.Draw.x) discard;
#if defined(EVANS_VISIBILITY_MESH)
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
    vec3 bary=gl_BaryCoordEXT;
#else
    vec2 screen=gl_FragCoord.xy*Parameters.Viewport.zw*vec2(2.0,-2.0)+vec2(-1.0,1.0);
    precise vec3 h=MeshPlaneX*screen.x+MeshPlaneY*screen.y+MeshPlaneOrigin;
    float denominator=h.x+h.y+h.z;
    if (denominator==0.0 || any(isnan(h)) || any(isinf(h))) discard;
    vec3 bary=h/denominator;
#endif
    if (any(isnan(bary)) || any(isinf(bary))) discard;
    vec2 RasterBary=bary.yz;
    if (Parameters.Material.z>0.0)
    {
#if defined(EVANS_MESH_NATIVE_BARYCENTRICS)
        uint safeTriangle=min(RasterReference-1u,max(Parameters.Draw.x,1u)-1u); Triangle item=Geometry.Records[safeTriangle];
        vec2 uv=CornerUV(item,0u)*bary.x+CornerUV(item,1u)*bary.y+CornerUV(item,2u)*bary.z;
        float alpha=dot(vec3(item.Color[0].a,item.Color[1].a,item.Color[2].a),bary)*Parameters.Material.y;
#else
        vec2 uv=MeshCoverageUV; float alpha=MeshCoverageAlpha*Parameters.Material.y;
#endif
        if (Parameters.Options.x!=0u) alpha*=texture(DiffuseMap,uv).a;
        if (Parameters.Options.z!=0u) alpha*=texture(OpacityMap,uv).r;
        if (clamp(alpha,0.0,1.0)<Parameters.Material.z) discard;
    }
#else
    float alpha = RasterOpacity * Parameters.Material.y;
    if (Parameters.Options.x != 0u) alpha *= texture(DiffuseMap, RasterUV).a;
    if (Parameters.Options.z != 0u) alpha *= texture(OpacityMap, RasterUV).r;
    if (Parameters.Material.z > 0.0 && clamp(alpha, 0.0, 1.0) < Parameters.Material.z) discard;
#endif
    // Rasterize the same triangle depth as the fill; keep a pixel-width edge.
    if (Parameters.Material.w > 1.5)
    {
        vec3 wireBary = vec3(1.0 - RasterBary.x - RasterBary.y, RasterBary);
        vec3 wireDx = dFdx(wireBary), wireDy = dFdy(wireBary);
        vec3 wireDistance = wireBary / max(sqrt(wireDx * wireDx + wireDy * wireDy), vec3(1.0e-8));
        if (min(wireDistance.x, min(wireDistance.y, wireDistance.z)) > 0.75) discard;
    }
    VisibilityID = RasterReference | (gl_FrontFacing ? 0u : BackFace); VisibilityBary = RasterBary;
}
#else
layout(location = 0) out vec4 DiffuseOpacity;
layout(location = 1) out vec4 NormalRoughness;
layout(location = 2) out vec4 SpecularShininess;
layout(location = 3) out vec4 AmbientMetallic;
layout(location = 4) out vec4 EmissionColor;
bool Derivatives(Triangle triangle, vec2 pixel, out vec3 bx, out vec3 by)
{
    bx = by = vec3(0.0);
    vec3 a = (Parameters.Camera * triangle.Position[0]).xyw;
    vec3 b = (Parameters.Camera * triangle.Position[1]).xyw;
    vec3 c = (Parameters.Camera * triangle.Position[2]).xyw;
    vec3 bound = max(abs(a), max(abs(b), abs(c))); float scale = max(bound.x, max(bound.y, bound.z));
    if (scale == 0.0) return false;
    a /= scale; b /= scale; c /= scale;
    vec3 row0 = cross(b, c), row1 = cross(c, a), row2 = cross(a, b);
    float determinant = dot(a, row0);
    if (abs(determinant) <= InverseEpsilon) return false;
    row0 /= determinant; row1 /= determinant; row2 /= determinant;
    vec3 point = vec3(pixel * Parameters.Viewport.zw * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 1.0);
    vec3 h = vec3(dot(row0, point), dot(row1, point), dot(row2, point)); float denominator = h.x + h.y + h.z;
    if (abs(denominator) <= InverseEpsilon) return false;
    vec3 hx = vec3(row0.x, row1.x, row2.x) * 2.0 * Parameters.Viewport.z;
    vec3 hy = vec3(row0.y, row1.y, row2.y) * -2.0 * Parameters.Viewport.w;
    bx = (hx - h / denominator * (hx.x + hx.y + hx.z)) / denominator;
    by = (hy - h / denominator * (hy.x + hy.y + hy.z)) / denominator;
    if (!any(isnan(bx)) && !any(isinf(bx)) && !any(isnan(by)) && !any(isinf(by))) return true;
    bx = by = vec3(0.0); return false;
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy); uint encoded = texelFetch(References, pixel, 0).r;
    uint reference = encoded & ReferenceMask;
    // Discard does not justify an invalid storage-buffer access in helper lanes.
    uint safeTriangle = min(reference > 0u ? reference - 1u : 0u, max(Parameters.Draw.x, 1u) - 1u);
    if (reference == 0u || reference > Parameters.Draw.x) discard;
    Triangle triangle = Geometry.Records[safeTriangle];
    uint owner=triangle.Source.x;
    if (Parameters.MaterialTiles.w!=0u) owner=MaterialBins[owner];
    if (owner!=(Parameters.MaterialTiles.w!=0u?Parameters.MaterialTiles.x:Parameters.Draw.w)) discard;
    vec2 stored = texelFetch(Barycentrics, pixel, 0).xy; vec3 bary = vec3(1.0 - stored.x - stored.y, stored);
    vec3 bx, by; bool gradients = Derivatives(triangle, gl_FragCoord.xy, bx, by);
    vec2 a = CornerUV(triangle, 0u), b = CornerUV(triangle, 1u), c = CornerUV(triangle, 2u);
    vec2 uv = a * bary.x + b * bary.y + c * bary.z;
    // Derivatives sum to zero: edge form avoids cancellation against the world origin.
    vec2 ux = (b - a) * bx.y + (c - a) * bx.z, uy = (b - a) * by.y + (c - a) * by.z;
    vec4 color = triangle.Color[0] * bary.x + triangle.Color[1] * bary.y + triangle.Color[2] * bary.z;
    vec4 texel = Parameters.Options.x != 0u ? textureGrad(DiffuseMap, uv, ux, uy) : vec4(1.0);
    if (float(Parameters.Options.w) == ToonShadingModel && Parameters.Options.x != 0u)
        texel.rgb = ToonTextureColor(DiffuseMap, uv, ux, uy);
    vec3 surfaceMap = Parameters.Options.z != 0 ? textureGrad(OpacityMap, uv, ux, uy).rgb : vec3(1, 1, 1);
    vec2 roughnessMetallic = Parameters.Surface.xy;
    if (Parameters.Options.z > 1) roughnessMetallic *= surfaceMap.gb;
    float opacity = surfaceMap.r;
    vec3 normal = Unit(triangle.Normal[0].xyz * bary.x + triangle.Normal[1].xyz * bary.y + triangle.Normal[2].xyz * bary.z, vec3(0.0, 1.0, 0.0));
    if (Parameters.Options.w == FlatModel)
    {
        vec3 face = Unit(cross(triangle.Position[1].xyz - triangle.Position[0].xyz, triangle.Position[2].xyz - triangle.Position[0].xyz), normal);
        normal = dot(face, normal) < 0.0 ? -face : face;
    }
    if ((encoded & BackFace) != 0u) normal = -normal;
    float uvDeterminant = ux.x * uy.y - ux.y * uy.x;
    if (gradients && Parameters.Options.y != 0u && Parameters.Options.w != FlatModel && abs(uvDeterminant) > TangentEpsilon)
    {
        vec3 px = (triangle.Position[1].xyz - triangle.Position[0].xyz) * bx.y + (triangle.Position[2].xyz - triangle.Position[0].xyz) * bx.z;
        vec3 py = (triangle.Position[1].xyz - triangle.Position[0].xyz) * by.y + (triangle.Position[2].xyz - triangle.Position[0].xyz) * by.z;
        vec3 tangent = (px * uy.y - py * ux.y) / uvDeterminant;
        vec3 sourceB = (py * ux.x - px * uy.x) / uvDeterminant;
        vec3 helper = abs(normal.y) < ParallelAxisThreshold ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = Unit(tangent - normal * dot(normal, tangent), Unit(cross(helper, normal), vec3(1.0, 0.0, 0.0)));
        vec3 bitangent = Unit(cross(normal, tangent), vec3(0.0, 0.0, 1.0));
        if (dot(bitangent, sourceB) < 0.0) bitangent = -bitangent;
        vec3 mapped = textureGrad(NormalMap, uv, ux, uy).xyz * 2.0 - 1.0;
        if (float(Parameters.Options.w) == ToonShadingModel) mapped = mix(vec3(0.0, 0.0, 1.0), mapped, ToonNormalStrength);
        normal = Unit(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z, normal);
    }
    vec3 surface = ToonAlbedo(color.rgb * max(Parameters.Tint.rgb, vec3(0.0)) * max(texel.rgb, vec3(0.0)), float(Parameters.Options.w));
    float metallic = clamp(roughnessMetallic.y, 0.0, 1.0);
    vec3 ambient = ToonAmbient(surface * Parameters.Diffuse.rgb, surface * max(Parameters.Ambient.rgb, vec3(0.0)) * max(Parameters.AmbientLight.rgb, vec3(0.0)) * max(Parameters.AmbientLight.w, 0.0), float(Parameters.Options.w));
    if (Parameters.Options.w == PbrModel) ambient *= 1.0 - metallic;
    DiffuseOpacity = vec4(clamp(surface * Parameters.Diffuse.rgb, 0.0, HalfFloatMaximum), clamp(color.a * Parameters.Material.y * texel.a * opacity, 0.0, 1.0));
    NormalRoughness = vec4(normal, clamp(roughnessMetallic.x, RoughnessMinimum, 1.0));
    SpecularShininess = clamp(vec4(Parameters.Specular.rgb, max(Parameters.Material.x, 1.0)), 0.0, HalfFloatMaximum);
    AmbientMetallic = vec4(clamp(ambient, 0.0, HalfFloatMaximum), metallic);
    EmissionColor = vec4(clamp(Parameters.Emission.rgb, 0.0, HalfFloatMaximum), 1.0);
    if (Parameters.Material.w > 0.0) { NormalRoughness.w = -1.0; EmissionColor.rgb = DecodeWire(Parameters.Tint.rgb); }
}
#endif
#endif
