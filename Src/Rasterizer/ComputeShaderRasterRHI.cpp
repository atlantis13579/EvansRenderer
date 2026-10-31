#include "Rasterizer/ComputeShaderRasterRHI.h"

#include "BackendCommon.hpp"
#include "GpuRasterCommon.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#endif

namespace Evans::Rasterizer
{
namespace
{

struct GpuVertex
{
    float Clip[4]{};
    float Color[4]{};
    float Position[4]{};
    float Normal[4]{};
    float ShadowClip[4]{};
    float TexCoord[4]{};
};

struct GpuTriangle
{
    GpuVertex Vertices[3];
    float BaseColor[4]{};
    float SpecularColor[4]{};
    float EmissiveColor[4]{};
    float Shininess = 1.0f;
    float AlphaCutoff = 0.0f;
    std::uint32_t DiffuseTextureIndex = InvalidRasterTextureIndex;
    std::uint32_t OpacityTextureIndex = InvalidRasterTextureIndex;
    std::uint32_t Selected = 0;
    std::uint32_t Padding[3]{};
};

struct alignas(16) GpuLight
{
    float PositionType[4]{};
    float DirectionIntensity[4]{};
    float ColorCastShadows[4]{};
    float AreaSizePadding[4]{};
};

struct alignas(16) GpuConstants
{
    float World[16]{};
    float View[16]{};
    float Projection[16]{};
    float LightViewProjection[16]{};
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    std::uint32_t TriangleCount = 0;
    std::uint32_t InputSpace = 0;
    std::uint32_t DepthRange = 0;
    std::uint32_t FillModeValue = 0;
    std::uint32_t ShadingValue = 0;
    std::uint32_t BackFaceCulling = 0;
    std::uint32_t FrontFaceCounterClockwise = 0;
    std::uint32_t OutlineEnabled = 0;
    std::uint32_t ShadowsEnabled = 0;
    std::uint32_t ShadowWidth = 0;
    std::uint32_t ShadowHeight = 0;
    std::uint32_t TilesX = 0;
    float WireframeWidth = 1.0f;
    float OutlineWidth = 1.0f;
    float ShadowBias = 0.001f;
    float ShadowStrength = 0.55f;
    std::uint32_t DepthTest = 1;
    std::uint32_t ShadowPCFRadius = 1;
    float ClearColor[4]{};
    float WireframeColor[4]{};
    float OutlineColor[4]{};
    float CameraPosition[4]{};
    float LightDirectionIntensity[4]{};
    float LightColorAmbientStrength[4]{};
    float AmbientColor[4]{};
    std::uint32_t LightCount = 0;
    std::int32_t ShadowLightIndex = -1;
    std::uint32_t TextureCount = 0;
    std::uint32_t TextureTexelCount = 0;
    GpuLight Lights[MaximumRasterLights]{};
    float ShadowRasterBias[4]{};
};

static_assert(sizeof(GpuVertex) == 96, "Compute raster vertex ABI mismatch");
static_assert(sizeof(GpuTriangle) == 368, "Compute raster triangle ABI mismatch");
static_assert(sizeof(GpuLight) == 64, "Compute raster light ABI mismatch");
static_assert(sizeof(GpuConstants) == 736, "Compute raster constants ABI mismatch");
static_assert(alignof(GpuConstants) == 16, "Compute raster constants alignment mismatch");
static_assert(
    offsetof(GpuLight, DirectionIntensity) == 16 &&
        offsetof(GpuLight, ColorCastShadows) == 32 &&
        offsetof(GpuLight, AreaSizePadding) == 48,
    "Compute raster light field offsets mismatch");
static_assert(
    offsetof(GpuVertex, TexCoord) == 80 &&
        offsetof(GpuTriangle, BaseColor) == 288 &&
        offsetof(GpuTriangle, Shininess) == 336 &&
        offsetof(GpuTriangle, Selected) == 352 &&
        offsetof(GpuConstants, LightCount) == 448 &&
        offsetof(GpuConstants, TextureCount) == 456 &&
        offsetof(GpuConstants, Lights) == 464 &&
        offsetof(GpuConstants, ShadowRasterBias) == 720,
    "Compute raster buffer offsets mismatch");

void CopyColor(float (&destination)[4], const Color& source) noexcept
{
    destination[0] = source.R;
    destination[1] = source.G;
    destination[2] = source.B;
    destination[3] = source.A;
}

void CopyMatrix(float (&destination)[16], const Matrix4& source) noexcept
{
    std::copy(source.Values.begin(), source.Values.end(), destination);
}

void CopyLight(GpuLight& destination, const RasterLight& source) noexcept
{
    destination.PositionType[0] = source.Position.X;
    destination.PositionType[1] = source.Position.Y;
    destination.PositionType[2] = source.Position.Z;
    destination.PositionType[3] = static_cast<float>(source.Type);
    destination.DirectionIntensity[0] = source.Direction.X;
    destination.DirectionIntensity[1] = source.Direction.Y;
    destination.DirectionIntensity[2] = source.Direction.Z;
    destination.DirectionIntensity[3] = source.Intensity;
    destination.ColorCastShadows[0] = source.LightColor.R;
    destination.ColorCastShadows[1] = source.LightColor.G;
    destination.ColorCastShadows[2] = source.LightColor.B;
    destination.ColorCastShadows[3] = source.CastShadows ? 1.0f : 0.0f;
    destination.AreaSizePadding[0] = source.AreaSize.X;
    destination.AreaSizePadding[1] = source.AreaSize.Y;
}

[[nodiscard]] std::vector<GpuTriangle> BuildGpuTriangles(const RasterScene& scene)
{
    std::vector<GpuTriangle> result(scene.Triangles.size());
    for (std::size_t triangleIndex = 0; triangleIndex < scene.Triangles.size(); ++triangleIndex)
    {
        const RasterTriangle& source = scene.Triangles[triangleIndex];
        GpuTriangle& destination = result[triangleIndex];
        for (std::size_t vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            const RasterVertex& vertex = source.Vertices[vertexIndex];
            GpuVertex& gpuVertex = destination.Vertices[vertexIndex];
            const float clip[] = {
                vertex.ClipPosition.X,
                vertex.ClipPosition.Y,
                vertex.ClipPosition.Z,
                vertex.ClipPosition.W};
            const float color[] = {
                vertex.VertexColor.R,
                vertex.VertexColor.G,
                vertex.VertexColor.B,
                vertex.VertexColor.A};
            const float position[] = {
                vertex.Position.X, vertex.Position.Y, vertex.Position.Z, 1.0f};
            const float normal[] = {
                vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z, 0.0f};
            const float shadowClip[] = {
                vertex.ShadowClipPosition.X,
                vertex.ShadowClipPosition.Y,
                vertex.ShadowClipPosition.Z,
                vertex.ShadowClipPosition.W};
            const float texCoord[] = {
                vertex.TexCoord.X, vertex.TexCoord.Y, 0.0f, 0.0f};
            std::copy(std::begin(clip), std::end(clip), gpuVertex.Clip);
            std::copy(std::begin(color), std::end(color), gpuVertex.Color);
            std::copy(std::begin(position), std::end(position), gpuVertex.Position);
            std::copy(std::begin(normal), std::end(normal), gpuVertex.Normal);
            std::copy(std::begin(shadowClip), std::end(shadowClip), gpuVertex.ShadowClip);
            std::copy(std::begin(texCoord), std::end(texCoord), gpuVertex.TexCoord);
        }
        CopyColor(destination.BaseColor, source.Material.BaseColor);
        CopyColor(destination.SpecularColor, source.Material.SpecularColor);
        CopyColor(destination.EmissiveColor, source.Material.EmissiveColor);
        destination.Shininess = source.Material.Shininess;
        destination.AlphaCutoff = source.Material.AlphaCutoff;
        destination.DiffuseTextureIndex = source.Material.DiffuseTextureIndex;
        destination.OpacityTextureIndex = source.Material.OpacityTextureIndex;
        destination.Selected = source.Selected ? 1U : 0U;
    }
    return result;
}

[[nodiscard]] GpuConstants BuildConstants(
    const RasterScene& scene,
    const RasterSettings& settings,
    const RenderTarget& target,
    bool shadowsEnabled,
    std::uint32_t shadowWidth,
    std::uint32_t shadowHeight,
    std::uint32_t textureCount,
    std::uint32_t textureTexelCount) noexcept
{
    GpuConstants constants;
    CopyMatrix(constants.World, settings.World);
    CopyMatrix(constants.View, settings.View);
    CopyMatrix(constants.Projection, settings.Projection);
    CopyMatrix(constants.LightViewProjection, settings.LightViewProjection);
    constants.Width = target.Width();
    constants.Height = target.Height();
    constants.TriangleCount = static_cast<std::uint32_t>(scene.Triangles.size());
    constants.InputSpace = static_cast<std::uint32_t>(settings.InputSpace);
    constants.DepthRange = static_cast<std::uint32_t>(settings.DepthRange);
    constants.FillModeValue = static_cast<std::uint32_t>(settings.Mode);
    constants.ShadingValue = static_cast<std::uint32_t>(settings.Shading);
    constants.BackFaceCulling = settings.BackFaceCulling ? 1U : 0U;
    constants.FrontFaceCounterClockwise = settings.FrontFaceCounterClockwise ? 1U : 0U;
    constants.OutlineEnabled = settings.OutlineEnabled ? 1U : 0U;
    constants.ShadowsEnabled = shadowsEnabled ? 1U : 0U;
    constants.ShadowWidth = shadowWidth;
    constants.ShadowHeight = shadowHeight;
    constants.TilesX = (target.Width() + 15U) / 16U;
    constants.WireframeWidth = settings.WireframeWidth;
    constants.OutlineWidth = settings.OutlineWidth;
    constants.ShadowBias = settings.ShadowBias;
    constants.ShadowRasterBias[0] = std::max(settings.ShadowSlopeScaledBias, 0.0f);
    constants.ShadowStrength = settings.ShadowStrength;
    constants.DepthTest = settings.DepthTest ? 1U : 0U;
    constants.ShadowPCFRadius = settings.ShadowPCFRadius;
    CopyColor(constants.ClearColor, settings.ClearColor);
    CopyColor(constants.WireframeColor, settings.WireframeColor);
    CopyColor(constants.OutlineColor, settings.OutlineColor);
    constants.CameraPosition[0] = settings.CameraPosition.X;
    constants.CameraPosition[1] = settings.CameraPosition.Y;
    constants.CameraPosition[2] = settings.CameraPosition.Z;
    constants.CameraPosition[3] = 1.0f;
    constants.LightDirectionIntensity[0] = settings.Light.DirectionToLight.X;
    constants.LightDirectionIntensity[1] = settings.Light.DirectionToLight.Y;
    constants.LightDirectionIntensity[2] = settings.Light.DirectionToLight.Z;
    constants.LightDirectionIntensity[3] = settings.Light.Intensity;
    constants.LightColorAmbientStrength[0] = settings.Light.LightColor.R;
    constants.LightColorAmbientStrength[1] = settings.Light.LightColor.G;
    constants.LightColorAmbientStrength[2] = settings.Light.LightColor.B;
    constants.LightColorAmbientStrength[3] = settings.AmbientStrength;
    CopyColor(constants.AmbientColor, settings.AmbientColor);
    constants.LightCount = EffectiveRasterLightCount(settings);
    constants.ShadowLightIndex =
        settings.ShadowLightIndex >= 0 &&
                static_cast<std::uint32_t>(settings.ShadowLightIndex) < constants.LightCount
            ? settings.ShadowLightIndex
            : -1;
    constants.TextureCount = textureCount;
    constants.TextureTexelCount = textureTexelCount;
    for (std::uint32_t index = 0; index < constants.LightCount; ++index)
    {
        CopyLight(constants.Lights[index], EffectiveRasterLight(settings, index));
    }
    return constants;
}

struct TileBins
{
    std::vector<std::uint32_t> Offsets;
    std::vector<std::uint32_t> TriangleIndices;
    std::uint32_t TilesX = 0;
    std::uint32_t TilesY = 0;
    std::uint32_t MaximumListLength = 0;
    std::uint64_t BinnedTriangles = 0;
    bool Cancelled = false;
};

[[nodiscard]] bool IsCancelled(
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation) noexcept
{
    return externalCancellation.IsCancellationRequested() ||
           backendCancellation.load(std::memory_order_relaxed);
}

[[nodiscard]] Float4 TransformForBinning(
    const RasterVertex& vertex,
    const RasterSettings& settings,
    bool lightPass) noexcept
{
    if (settings.InputSpace == VertexInputSpace::Clip)
    {
        return lightPass ? vertex.ShadowClipPosition : vertex.ClipPosition;
    }
    const Float4 local{vertex.Position.X, vertex.Position.Y, vertex.Position.Z, 1.0f};
    const Float4 world = settings.World * local;
    return lightPass
               ? settings.LightViewProjection * world
               : settings.Projection * (settings.View * world);
}

[[nodiscard]] TileBins BuildTileBins(
    const RasterScene& scene,
    const RasterSettings& settings,
    std::uint32_t width,
    std::uint32_t height,
    bool lightPass,
    const RasterCancellationToken& externalCancellation,
    const std::atomic<bool>& backendCancellation)
{
    constexpr std::uint32_t tileSize = 16U;
    struct Bounds
    {
        int MinimumX = 0;
        int MinimumY = 0;
        int MaximumX = -1;
        int MaximumY = -1;
        bool Valid = false;
    };

    TileBins result;
    result.TilesX = (width + tileSize - 1U) / tileSize;
    result.TilesY = (height + tileSize - 1U) / tileSize;
    const std::uint32_t tileCount = result.TilesX * result.TilesY;
    std::vector<std::uint32_t> counts(tileCount, 0U);
    std::vector<Bounds> triangleBounds(scene.Triangles.size());
    for (std::size_t triangleIndex = 0; triangleIndex < scene.Triangles.size(); ++triangleIndex)
    {
        if ((triangleIndex & 255U) == 0U &&
            IsCancelled(externalCancellation, backendCancellation))
        {
            result.Cancelled = true;
            return result;
        }

        float screenX[3]{};
        float screenY[3]{};
        bool valid = true;
        for (std::size_t vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            const Float4 clip = TransformForBinning(
                scene.Triangles[triangleIndex].Vertices[vertexIndex],
                settings,
                lightPass);
            if (!std::isfinite(clip.X) || !std::isfinite(clip.Y) ||
                !std::isfinite(clip.W) || clip.W <= 1.0e-6f)
            {
                valid = false;
                break;
            }
            const float inverseW = 1.0f / clip.W;
            screenX[vertexIndex] =
                (clip.X * inverseW * 0.5f + 0.5f) * static_cast<float>(width);
            screenY[vertexIndex] =
                (0.5f - clip.Y * inverseW * 0.5f) * static_cast<float>(height);
        }
        if (!valid)
        {
            continue;
        }

        const float padding = !lightPass && settings.Mode != FillMode::Solid
            ? std::max(settings.WireframeWidth, 1.0f) * 0.5f : 0.0f;
        Bounds bounds;
        bounds.MinimumX = std::max(
            0,
            static_cast<int>(std::floor(
                std::min({screenX[0], screenX[1], screenX[2]}) - padding)));
        bounds.MinimumY = std::max(
            0,
            static_cast<int>(std::floor(
                std::min({screenY[0], screenY[1], screenY[2]}) - padding)));
        bounds.MaximumX = std::min(
            static_cast<int>(width) - 1,
            static_cast<int>(std::ceil(
                std::max({screenX[0], screenX[1], screenX[2]}) + padding)));
        bounds.MaximumY = std::min(
            static_cast<int>(height) - 1,
            static_cast<int>(std::ceil(
                std::max({screenY[0], screenY[1], screenY[2]}) + padding)));
        bounds.Valid = bounds.MinimumX <= bounds.MaximumX &&
                       bounds.MinimumY <= bounds.MaximumY;
        triangleBounds[triangleIndex] = bounds;
        if (!bounds.Valid)
        {
            continue;
        }
        ++result.BinnedTriangles;
        for (std::uint32_t tileY =
                 static_cast<std::uint32_t>(bounds.MinimumY) / tileSize;
             tileY <= static_cast<std::uint32_t>(bounds.MaximumY) / tileSize;
             ++tileY)
        {
            for (std::uint32_t tileX =
                     static_cast<std::uint32_t>(bounds.MinimumX) / tileSize;
                 tileX <= static_cast<std::uint32_t>(bounds.MaximumX) / tileSize;
                 ++tileX)
            {
                ++counts[tileY * result.TilesX + tileX];
            }
        }
    }

    result.Offsets.resize(static_cast<std::size_t>(tileCount) + 1U, 0U);
    for (std::uint32_t tile = 0; tile < tileCount; ++tile)
    {
        result.MaximumListLength = std::max(result.MaximumListLength, counts[tile]);
        result.Offsets[tile + 1U] = result.Offsets[tile] + counts[tile];
    }
    result.TriangleIndices.resize(result.Offsets.back());
    std::vector<std::uint32_t> writeOffsets = result.Offsets;
    for (std::uint32_t triangleIndex = 0;
         triangleIndex < static_cast<std::uint32_t>(triangleBounds.size());
         ++triangleIndex)
    {
        if ((triangleIndex & 255U) == 0U &&
            IsCancelled(externalCancellation, backendCancellation))
        {
            result.Cancelled = true;
            return result;
        }
        const Bounds& bounds = triangleBounds[triangleIndex];
        if (!bounds.Valid)
        {
            continue;
        }
        for (std::uint32_t tileY =
                 static_cast<std::uint32_t>(bounds.MinimumY) / tileSize;
             tileY <= static_cast<std::uint32_t>(bounds.MaximumY) / tileSize;
             ++tileY)
        {
            for (std::uint32_t tileX =
                     static_cast<std::uint32_t>(bounds.MinimumX) / tileSize;
                 tileX <= static_cast<std::uint32_t>(bounds.MaximumX) / tileSize;
                 ++tileX)
            {
                const std::uint32_t tile = tileY * result.TilesX + tileX;
                result.TriangleIndices[writeOffsets[tile]++] = triangleIndex;
            }
        }
    }
    return result;
}

constexpr const char* ComputeShaderSource = R"hlsl(
struct Vertex {
    float4 Clip; float4 Color; float4 Position; float4 Normal; float4 ShadowClip;
    float4 TexCoord;
};
struct RasterTriangleGpu {
    Vertex Vertices[3]; float4 BaseColor; float4 SpecularColor; float4 EmissiveColor;
    float Shininess; float AlphaCutoff; uint DiffuseTextureIndex; uint OpacityTextureIndex;
    uint Selected; uint Padding0; uint Padding1; uint Padding2;
};
struct RasterTextureGpu { uint OffsetTexels; uint Width; uint Height; uint Padding; };
struct RasterLightGpu {
    float4 PositionType; float4 DirectionIntensity;
    float4 ColorCastShadows; float4 AreaSizePadding;
};
cbuffer Constants : register(b0) {
    row_major float4x4 World; row_major float4x4 View; row_major float4x4 Projection;
    row_major float4x4 LightViewProjection;
    uint Width, Height, TriangleCount, InputSpace;
    uint DepthRange, FillModeValue, ShadingValue, BackFaceCulling;
    uint FrontFaceCounterClockwise, OutlineEnabled, ShadowsEnabled, ShadowWidth;
    uint ShadowHeight, TilesX; float WireframeWidth, OutlineWidth;
    float ShadowBias, ShadowStrength; uint DepthTest, ShadowPCFRadius;
    float4 ClearColor, WireframeColor, OutlineColor, CameraPosition;
    float4 LightDirectionIntensity, LightColorAmbientStrength, AmbientColor;
    uint LightCount; int ShadowLightIndex; uint TextureCount, TextureTexelCount;
    RasterLightGpu Lights[4];
    float4 ShadowRasterBias;
};
StructuredBuffer<RasterTriangleGpu> Triangles : register(t0);
StructuredBuffer<uint> TileOffsets : register(t1);
StructuredBuffer<uint> TileTriangleIndices : register(t2);
StructuredBuffer<float> ShadowDepths : register(t3);
StructuredBuffer<RasterTextureGpu> TextureDescriptors : register(t4);
StructuredBuffer<uint> TextureTexels : register(t5);
RWStructuredBuffer<float4> OutputColors : register(u0);
RWStructuredBuffer<float> OutputDepths : register(u1);
RWStructuredBuffer<float> GeneratedShadowDepths : register(u2);

float Edge(float2 a,float2 b,float2 p) {
    return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
}
float3 NormalizeSafe(float3 value) {
    float lengthSquared=dot(value,value);
    return lengthSquared<1.0e-12?float3(0,0,1):value*rsqrt(lengthSquared);
}
float3 NormalizeOr(float3 value,float3 fallback) {
    float lengthSquared=dot(value,value);
    return lengthSquared<1.0e-12?fallback:value*rsqrt(lengthSquared);
}
float SRGBToLinearChannel(float value) {
    value=saturate(value);
    return value<=0.04045?value/12.92:pow((value+0.055)/1.055,2.4);
}
int WrapTextureCoordinate(int value,uint size) {
    int result=value%(int)size;
    return result<0?result+(int)size:result;
}
float4 ReadTextureTexel(RasterTextureGpu descriptor,int x,int y,bool decodeSRGB) {
    x=WrapTextureCoordinate(x,descriptor.Width);
    y=WrapTextureCoordinate(y,descriptor.Height);
    uint packed=TextureTexels[descriptor.OffsetTexels+(uint)y*descriptor.Width+(uint)x];
    float4 value=float4(
        packed&255U,(packed>>8U)&255U,(packed>>16U)&255U,(packed>>24U)&255U)/255.0;
    if(decodeSRGB) {
        value.rgb=float3(
            SRGBToLinearChannel(value.r),
            SRGBToLinearChannel(value.g),
            SRGBToLinearChannel(value.b));
    }
    return value;
}
float4 SampleTexture(uint textureIndex,float2 texCoord,bool decodeSRGB) {
    if(textureIndex>=TextureCount)return 1.0.xxxx;
    RasterTextureGpu descriptor=TextureDescriptors[textureIndex];
    uint texelCount=descriptor.Width*descriptor.Height;
    if(descriptor.Width==0||descriptor.Height==0||descriptor.OffsetTexels>=TextureTexelCount||
       texelCount>TextureTexelCount-descriptor.OffsetTexels)return 1.0.xxxx;
    float x=frac(texCoord.x)*descriptor.Width-0.5;
    float y=frac(1.0-texCoord.y)*descriptor.Height-0.5;
    int x0=(int)floor(x),y0=(int)floor(y);
    float blendX=x-floor(x),blendY=y-floor(y);
    float4 top=lerp(
        ReadTextureTexel(descriptor,x0,y0,decodeSRGB),
        ReadTextureTexel(descriptor,x0+1,y0,decodeSRGB),blendX);
    float4 bottom=lerp(
        ReadTextureTexel(descriptor,x0,y0+1,decodeSRGB),
        ReadTextureTexel(descriptor,x0+1,y0+1,decodeSRGB),blendX);
    return lerp(top,bottom,blendY);
}
float4 SampleSurface(
    RasterTriangleGpu primitive,float p0,float p1,float p2,float denominator) {
    float4 vertex=(primitive.Vertices[0].Color*p0+
                   primitive.Vertices[1].Color*p1+
                   primitive.Vertices[2].Color*p2)/denominator;
    float2 texCoord=(primitive.Vertices[0].TexCoord.xy*p0+
                     primitive.Vertices[1].TexCoord.xy*p1+
                     primitive.Vertices[2].TexCoord.xy*p2)/denominator;
    float4 diffuse=SampleTexture(primitive.DiffuseTextureIndex,texCoord,true);
    float opacity=SampleTexture(primitive.OpacityTextureIndex,texCoord,false).r;
    return float4(
        vertex.rgb*primitive.BaseColor.rgb*diffuse.rgb,
        saturate(vertex.a*primitive.BaseColor.a*diffuse.a*opacity));
}
void AccumulateLightSample(
    float3 lightDirection,float3 radiance,float visibility,float3 normal,
    float3 viewDirection,float shininess,uint shading,
    inout float3 diffuseLight,inout float3 specularLight) {
    float directionLengthSquared=dot(lightDirection,lightDirection);
    if(directionLengthSquared<=1.0e-12)return;
    float3 direction=lightDirection*rsqrt(directionLengthSquared);
    float normalToLight=max(dot(normal,direction),0.0);
    if(normalToLight<=0.0)return;
    float3 visibleRadiance=radiance*visibility;
    diffuseLight+=visibleRadiance*normalToLight;
    if(shading==2) {
        float3 halfVector=NormalizeOr(direction+viewDirection,normal);
        float specularAmount=pow(max(dot(normal,halfVector),0.0),max(shininess,1.0));
        specularLight+=visibleRadiance*specularAmount;
    }
}
void AccumulateSceneLight(
    uint lightIndex,float3 position,float3 normal,float3 viewDirection,
    float shininess,uint shading,float shadowVisibility,
    inout float3 diffuseLight,inout float3 specularLight) {
    RasterLightGpu light=Lights[lightIndex];
    uint type=(uint)(light.PositionType.w+0.5);
    float visibility=(int(lightIndex)==ShadowLightIndex&&light.ColorCastShadows.w>0.5)
        ?shadowVisibility:1.0;
    float3 radiance=max(light.ColorCastShadows.rgb,0.0)*
        max(light.DirectionIntensity.w,0.0);
    if(type==0) {
        AccumulateLightSample(light.DirectionIntensity.xyz,radiance,visibility,normal,
            viewDirection,shininess,shading,diffuseLight,specularLight);
        return;
    }
    if(type==1) {
        float3 toLight=light.PositionType.xyz-position;
        float distanceSquared=max(dot(toLight,toLight),1.0e-6);
        AccumulateLightSample(toLight,radiance/distanceSquared,visibility,normal,
            viewDirection,shininess,shading,diffuseLight,specularLight);
        return;
    }
    float3 emissionDirection=NormalizeOr(
        light.DirectionIntensity.xyz,float3(0,-1,0));
    float3 helper=abs(emissionDirection.y)<0.999?float3(0,1,0):float3(1,0,0);
    float3 tangent=NormalizeOr(cross(helper,emissionDirection),float3(1,0,0));
    float3 bitangent=NormalizeOr(cross(emissionDirection,tangent),float3(0,0,1));
    [unroll]for(int sampleY=0;sampleY<2;++sampleY) {
        [unroll]for(int sampleX=0;sampleX<2;++sampleX) {
            float offsetX=sampleX==0?-0.25:0.25;
            float offsetY=sampleY==0?-0.25:0.25;
            float3 samplePosition=light.PositionType.xyz+
                tangent*(offsetX*light.AreaSizePadding.x)+
                bitangent*(offsetY*light.AreaSizePadding.y);
            float3 toLight=samplePosition-position;
            float distanceSquared=max(dot(toLight,toLight),1.0e-6);
            float3 direction=toLight*rsqrt(distanceSquared);
            float emitterCosine=max(dot(emissionDirection,-direction),0.0);
            AccumulateLightSample(direction,
                radiance*(0.25*emitterCosine/distanceSquared),visibility,normal,
                viewDirection,shininess,shading,diffuseLight,specularLight);
        }
    }
}
float4 LightClip(Vertex vertex) {
    return InputSpace==0?vertex.ShadowClip:
        mul(LightViewProjection,mul(World,vertex.Position));
}

[numthreads(8,8,1)]
void ShadowMain(uint3 id:SV_DispatchThreadID) {
    uint x=id.x,y=id.y;if(x>=ShadowWidth||y>=ShadowHeight)return;
    uint pixel=y*ShadowWidth+x;
    uint tile=(y/16U)*((ShadowWidth+15U)/16U)+x/16U;
    float2 samplePosition=float2(x+0.5,y+0.5);float bestDepth=1.0e30;
    [loop]for(uint binIndex=TileOffsets[tile];binIndex<TileOffsets[tile+1U];++binIndex) {
        RasterTriangleGpu primitive=Triangles[TileTriangleIndices[binIndex]];
        float2 screen[3];float inverseW[3],vertexDepth[3];bool valid=true;
        [unroll]for(uint vertexIndex=0;vertexIndex<3;++vertexIndex) {
            float4 clip=LightClip(primitive.Vertices[vertexIndex]);
            if(clip.w<=1.0e-6){valid=false;break;}
            inverseW[vertexIndex]=rcp(clip.w);float3 ndc=clip.xyz*inverseW[vertexIndex];
            screen[vertexIndex]=float2(
                (ndc.x*0.5+0.5)*ShadowWidth,(0.5-ndc.y*0.5)*ShadowHeight);
            vertexDepth[vertexIndex]=DepthRange==0?ndc.z:ndc.z*0.5+0.5;
        }
        if(!valid)continue;
        float area=Edge(screen[0],screen[1],screen[2]);if(abs(area)<1.0e-6)continue;
        float w0=Edge(screen[1],screen[2],samplePosition)/area;
        float w1=Edge(screen[2],screen[0],samplePosition)/area;
        float w2=1.0-w0-w1;if(w0<0||w1<0||w2<0)continue;
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1)continue;
        if(ShadowRasterBias.x>0) {
            float dz1=vertexDepth[1]-vertexDepth[0],dz2=vertexDepth[2]-vertexDepth[0];
            float slopeX=(dz1*(screen[2].y-screen[0].y)-dz2*(screen[1].y-screen[0].y))/area;
            float slopeY=((screen[1].x-screen[0].x)*dz2-(screen[2].x-screen[0].x)*dz1)/area;
            depth=min(depth+ShadowRasterBias.x*max(abs(slopeX),abs(slopeY)),1.0);
        }
        if(depth>=bestDepth)continue;
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2];
        float denominator=p0+p1+p2;if(abs(denominator)<1.0e-8)continue;
        float alpha=SampleSurface(primitive,p0,p1,p2,denominator).a;
        if(primitive.AlphaCutoff>0.0&&alpha<primitive.AlphaCutoff)continue;
        bestDepth=depth;
    }
    GeneratedShadowDepths[pixel]=bestDepth;
}

)hlsl"
R"hlsl(
[numthreads(8,8,1)]
void RasterMain(uint3 id:SV_DispatchThreadID) {
    uint x=id.x,y=id.y;if(x>=Width||y>=Height)return;
    uint pixel=y*Width+x,tile=(y/16U)*TilesX+x/16U;
    float2 samplePosition=float2(x+0.5,y+0.5);
    float bestDepth=1.0e30,bestWireDepth=1.0e30;float4 bestColor=ClearColor;
    [loop]for(uint binIndex=TileOffsets[tile];binIndex<TileOffsets[tile+1U];++binIndex) {
        RasterTriangleGpu primitive=Triangles[TileTriangleIndices[binIndex]];
        float4 clip[3],world[3],shadowClip[3];float3 normal[3];
        float2 screen[3];float inverseW[3],vertexDepth[3];bool valid=true;
        [unroll]for(uint vertexIndex=0;vertexIndex<3;++vertexIndex) {
            Vertex vertex=primitive.Vertices[vertexIndex];
            if(InputSpace==0) {
                clip[vertexIndex]=vertex.Clip;world[vertexIndex]=vertex.Position;
                normal[vertexIndex]=vertex.Normal.xyz;shadowClip[vertexIndex]=vertex.ShadowClip;
            } else {
                world[vertexIndex]=mul(World,vertex.Position);
                clip[vertexIndex]=mul(Projection,mul(View,world[vertexIndex]));
                normal[vertexIndex]=NormalizeSafe(mul((float3x3)World,vertex.Normal.xyz));
                shadowClip[vertexIndex]=mul(LightViewProjection,world[vertexIndex]);
            }
            if(clip[vertexIndex].w<=1.0e-6){valid=false;break;}
            inverseW[vertexIndex]=rcp(clip[vertexIndex].w);
            float3 ndc=clip[vertexIndex].xyz*inverseW[vertexIndex];
            screen[vertexIndex]=float2((ndc.x*0.5+0.5)*Width,(0.5-ndc.y*0.5)*Height);
            vertexDepth[vertexIndex]=DepthRange==0?ndc.z:ndc.z*0.5+0.5;
        }
        if(!valid)continue;
        float area=Edge(screen[0],screen[1],screen[2]);if(abs(area)<1.0e-6)continue;
        bool front=FrontFaceCounterClockwise!=0?area<0:area>0;
        if(BackFaceCulling!=0&&!front)continue;
        float w0=Edge(screen[1],screen[2],samplePosition)/area;
        float w1=Edge(screen[2],screen[0],samplePosition)/area,w2=1.0-w0-w1;
        bool inside=w0>=0&&w1>=0&&w2>=0;
        bool wire=false;
        float3 edgeWeights=0.0.xxx;
        if(FillModeValue!=0){
            float nearestEdge=1.0e30;
            [unroll]for(int edgeIndex=0;edgeIndex<3;++edgeIndex){
                int next=(edgeIndex+1)%3;
                float2 segment=screen[next]-screen[edgeIndex];
                float t=saturate(dot(samplePosition-screen[edgeIndex],segment)/max(dot(segment,segment),1.0e-12));
                float2 offset=samplePosition-(screen[edgeIndex]+t*segment);
                float distanceSquared=dot(offset,offset);
                if(distanceSquared<nearestEdge){
                    nearestEdge=distanceSquared;edgeWeights=0.0.xxx;
                    edgeWeights[edgeIndex]=1.0-t;edgeWeights[next]=t;
                }
            }
            float halfWidth=max(WireframeWidth,1.0)*0.5;
            wire=nearestEdge<=halfWidth*halfWidth;
        }
        if(wire){
            float wireDepth=dot(edgeWeights,float3(vertexDepth[0],vertexDepth[1],vertexDepth[2]));
            float3 edgePerspective=edgeWeights*float3(inverseW[0],inverseW[1],inverseW[2]);
            float edgeDenominator=edgePerspective.x+edgePerspective.y+edgePerspective.z;
            if(wireDepth>=0&&wireDepth<=1&&abs(edgeDenominator)>1.0e-8){
                float edgeAlpha=SampleSurface(primitive,edgePerspective.x,edgePerspective.y,
                    edgePerspective.z,edgeDenominator).a;
                if(primitive.AlphaCutoff<=0||edgeAlpha>=primitive.AlphaCutoff)
                    bestWireDepth=min(bestWireDepth,wireDepth);
            }
        }
        if(FillModeValue==1||!inside)continue;
        float depth=w0*vertexDepth[0]+w1*vertexDepth[1]+w2*vertexDepth[2];
        if(depth<0||depth>1||(DepthTest!=0&&depth>=bestDepth))continue;
        float p0=w0*inverseW[0],p1=w1*inverseW[1],p2=w2*inverseW[2];
        float denominator=p0+p1+p2;if(abs(denominator)<1.0e-8)continue;
        float4 color=SampleSurface(primitive,p0,p1,p2,denominator);
        if(primitive.AlphaCutoff>0.0&&color.a<primitive.AlphaCutoff)continue;

        float shadowVisibility=1.0;
        if(ShadowsEnabled!=0&&ShadowWidth>0&&ShadowHeight>0) {
            float4 shadow=(shadowClip[0]*p0+shadowClip[1]*p1+shadowClip[2]*p2)/denominator;
            if(shadow.w>1.0e-6) {
                float inverseShadowW=rcp(shadow.w);
                float shadowX=shadow.x*inverseShadowW*0.5+0.5;
                float shadowY=0.5-shadow.y*inverseShadowW*0.5;
                float shadowZ=shadow.z*inverseShadowW;
                if(DepthRange!=0)shadowZ=shadowZ*0.5+0.5;
                if(shadowX>=0&&shadowX<=1&&shadowY>=0&&shadowY<=1&&shadowZ>=0&&shadowZ<=1) {
                    int centerX=min((int)(shadowX*ShadowWidth),(int)ShadowWidth-1);
                    int centerY=min((int)(shadowY*ShadowHeight),(int)ShadowHeight-1);
                    int radius=(int)min(ShadowPCFRadius,8U);
                    uint occluded=0,validSamples=0;
                    [loop]for(int offsetY=-radius;offsetY<=radius;++offsetY) {
                        int sy=centerY+offsetY;if(sy<0||sy>=(int)ShadowHeight)continue;
                        [loop]for(int offsetX=-radius;offsetX<=radius;++offsetX) {
                            int sx=centerX+offsetX;if(sx<0||sx>=(int)ShadowWidth)continue;
                            ++validSamples;
                            if(shadowZ-ShadowBias>ShadowDepths[sy*ShadowWidth+sx])++occluded;
                        }
                    }
                    if(validSamples>0)
                        shadowVisibility=1.0-saturate(ShadowStrength)*
                            ((float)occluded/(float)validSamples);
                }
            }
        }

        if(ShadingValue==0) {
            color.rgb+=primitive.EmissiveColor.rgb;
        } else {
            float3 n=NormalizeSafe((normal[0]*p0+normal[1]*p1+normal[2]*p2)/denominator);
            if(!front)n=-n;
            float3 position=((world[0]*p0+world[1]*p1+world[2]*p2)/denominator).xyz;
            float3 viewDirection=NormalizeOr(CameraPosition.xyz-position,n);
            float3 diffuseLight=0.0.xxx,specularLight=0.0.xxx;
            [loop]for(uint lightIndex=0;lightIndex<min(LightCount,4U);++lightIndex)
                AccumulateSceneLight(lightIndex,position,n,viewDirection,primitive.Shininess,
                    ShadingValue,shadowVisibility,diffuseLight,specularLight);
            color.rgb=color.rgb*(AmbientColor.rgb*LightColorAmbientStrength.w+diffuseLight)+
                primitive.SpecularColor.rgb*specularLight+primitive.EmissiveColor.rgb;
        }
        color.rgb=max(color.rgb,0.0);
        color.rgb/=max(1.0,max(color.r,max(color.g,color.b)));
        bestDepth=depth;bestColor=float4(saturate(color.rgb),saturate(color.a));
    }
    if(FillModeValue==1)bestDepth=bestWireDepth;
    if(FillModeValue!=0&&bestWireDepth<=1.0&&(DepthTest==0||bestWireDepth<=bestDepth+1.0e-4))
    {bestColor=WireframeColor;bestDepth=min(bestDepth,bestWireDepth);}
    OutputColors[pixel]=bestColor;OutputDepths[pixel]=bestDepth;
}
)hlsl";

#if defined(_WIN32)

template <typename Interface>
void Release(Interface*& value) noexcept
{
    if (value != nullptr)
    {
        value->Release();
        value = nullptr;
    }
}

class D3D11ComputeRuntime
{
public:
    ~D3D11ComputeRuntime() { Shutdown(); }

    [[nodiscard]] bool Initialize(std::string& error)
    {
        Shutdown();
        D3D11Library_ = LoadLibraryA("d3d11.dll");
        CompilerLibrary_ = LoadLibraryA("d3dcompiler_47.dll");
        if (D3D11Library_ == nullptr || CompilerLibrary_ == nullptr)
        {
            error = "D3D11 or D3DCompiler system library is unavailable";
            Shutdown();
            return false;
        }
        CreateDevice_ = reinterpret_cast<CreateDeviceFunction>(
            GetProcAddress(D3D11Library_, "D3D11CreateDevice"));
        Compile_ = reinterpret_cast<CompileFunction>(
            GetProcAddress(CompilerLibrary_, "D3DCompile"));
        if (CreateDevice_ == nullptr || Compile_ == nullptr)
        {
            error = "D3D11 system libraries are missing required entry points";
            Shutdown();
            return false;
        }

        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selectedLevel = D3D_FEATURE_LEVEL_11_0;
        const HRESULT deviceResult = CreateDevice_(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            levels,
            1,
            D3D11_SDK_VERSION,
            &Device_,
            &selectedLevel,
            &ImmediateContext_);
        if (FAILED(deviceResult) || selectedLevel < D3D_FEATURE_LEVEL_11_0)
        {
            error = "No Direct3D 11 hardware compute device is available (HRESULT " +
                    std::to_string(static_cast<unsigned long>(deviceResult)) + ")";
            Shutdown();
            return false;
        }

        const auto compileShader = [&](
            const char* entryPoint,
            ID3D11ComputeShader** output) -> bool
        {
            ID3DBlob* byteCode = nullptr;
            ID3DBlob* compilerErrors = nullptr;
            const HRESULT compileResult = Compile_(
                ComputeShaderSource,
                std::strlen(ComputeShaderSource),
                "EvansComputeRaster.hlsl",
                nullptr,
                nullptr,
                entryPoint,
                "cs_5_0",
                D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS,
                0,
                &byteCode,
                &compilerErrors);
            if (FAILED(compileResult))
            {
                error = std::string("Failed to compile D3D11 compute entry ") +
                        entryPoint;
                if (compilerErrors != nullptr)
                {
                    error += ": " + std::string(
                        static_cast<const char*>(compilerErrors->GetBufferPointer()),
                        compilerErrors->GetBufferSize());
                }
                Release(compilerErrors);
                Release(byteCode);
                return false;
            }
            Release(compilerErrors);
            const HRESULT shaderResult = Device_->CreateComputeShader(
                byteCode->GetBufferPointer(),
                byteCode->GetBufferSize(),
                nullptr,
                output);
            Release(byteCode);
            if (FAILED(shaderResult))
            {
                error = std::string("Failed to create D3D11 compute entry ") +
                        entryPoint;
                return false;
            }
            return true;
        };
        if (!compileShader("RasterMain", &RasterShader_) ||
            !compileShader("ShadowMain", &ShadowShader_))
        {
            Shutdown();
            return false;
        }
        return true;
    }

    void Shutdown() noexcept
    {
        Release(ShadowShader_);
        Release(RasterShader_);
        if (ImmediateContext_ != nullptr) ImmediateContext_->ClearState();
        Release(ImmediateContext_);
        Release(Device_);
        if (CompilerLibrary_ != nullptr) FreeLibrary(CompilerLibrary_);
        if (D3D11Library_ != nullptr) FreeLibrary(D3D11Library_);
        CompilerLibrary_ = nullptr;
        D3D11Library_ = nullptr;
        CreateDevice_ = nullptr;
        Compile_ = nullptr;
    }

    [[nodiscard]] RasterStatus Render(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& externalCancellation,
        const std::atomic<bool>& backendCancellation,
        RasterStatistics& statistics)
    {
        if (IsCancelled(externalCancellation, backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const bool hasExternalShadowMap =
            settings.ShadowsEnabled && settings.SceneShadowMap != nullptr &&
            settings.SceneShadowMap->IsValid();
        const bool generateShadowMap =
            settings.ShadowsEnabled && !hasExternalShadowMap &&
            settings.AutoGenerateShadowMap;
        const bool shadowsEnabled = hasExternalShadowMap || generateShadowMap;
        const std::uint32_t shadowWidth = hasExternalShadowMap
                                              ? settings.SceneShadowMap->Width
                                              : (generateShadowMap
                                                     ? std::clamp(
                                                           settings.ShadowMapSize,
                                                           16U,
                                                           4096U)
                                                     : 0U);
        const std::uint32_t shadowHeight = hasExternalShadowMap
                                               ? settings.SceneShadowMap->Height
                                               : shadowWidth;

        const auto transformBegin = std::chrono::steady_clock::now();
        Detail::PreparedGpuRasterScenes prepared = Detail::PrepareGpuRasterScenes(
            scene,
            settings,
            generateShadowMap,
            externalCancellation,
            backendCancellation);
        if (prepared.Cancelled ||
            IsCancelled(externalCancellation, backendCancellation))
        {
            return RasterStatus::Cancelled;
        }
        const RasterScene& mainScene = prepared.Main;
        const RasterScene& shadowScene = prepared.Shadow;
        const RasterSettings& gpuSettings = prepared.Settings;
        const TileBins mainBins = BuildTileBins(
            mainScene,
            gpuSettings,
            target.Width(),
            target.Height(),
            false,
            externalCancellation,
            backendCancellation);
        if (mainBins.Cancelled ||
            IsCancelled(externalCancellation, backendCancellation))
        {
            return RasterStatus::Cancelled;
        }
        const TileBins shadowBins = generateShadowMap
                                        ? BuildTileBins(
                                              shadowScene,
                                              gpuSettings,
                                              shadowWidth,
                                              shadowHeight,
                                              true,
                                              externalCancellation,
                                              backendCancellation)
                                        : TileBins{};
        if (shadowBins.Cancelled ||
            IsCancelled(externalCancellation, backendCancellation))
        {
            return RasterStatus::Cancelled;
        }
        const std::vector<GpuTriangle> triangles =
            BuildGpuTriangles(mainScene);
        const std::vector<GpuTriangle> shadowTriangles =
            generateShadowMap
                ? BuildGpuTriangles(shadowScene)
                : std::vector<GpuTriangle>{};
        const Detail::GpuRasterTextureData textures =
            Detail::BuildGpuRasterTextures(mainScene);
        if (IsCancelled(externalCancellation, backendCancellation))
        {
            return RasterStatus::Cancelled;
        }
        const GpuConstants constants = BuildConstants(
            mainScene,
            gpuSettings,
            target,
            shadowsEnabled,
            shadowWidth,
            shadowHeight,
            static_cast<std::uint32_t>(textures.Descriptors.size()),
            static_cast<std::uint32_t>(textures.TexelsRGBA8.size()));
        statistics.TransformMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - transformBegin)
                .count();

        const std::size_t pixelCount =
            static_cast<std::size_t>(target.Width()) * target.Height();
        const std::size_t shadowPixelCount = std::max<std::size_t>(
            static_cast<std::size_t>(shadowWidth) * shadowHeight,
            1U);
        ID3D11Buffer* triangleBuffer = nullptr;
        ID3D11ShaderResourceView* triangleView = nullptr;
        ID3D11Buffer* shadowTriangleBuffer = nullptr;
        ID3D11ShaderResourceView* shadowTriangleView = nullptr;
        ID3D11Buffer* mainOffsetBuffer = nullptr;
        ID3D11ShaderResourceView* mainOffsetView = nullptr;
        ID3D11Buffer* mainIndexBuffer = nullptr;
        ID3D11ShaderResourceView* mainIndexView = nullptr;
        ID3D11Buffer* shadowOffsetBuffer = nullptr;
        ID3D11ShaderResourceView* shadowOffsetView = nullptr;
        ID3D11Buffer* shadowIndexBuffer = nullptr;
        ID3D11ShaderResourceView* shadowIndexView = nullptr;
        ID3D11Buffer* shadowDepthBuffer = nullptr;
        ID3D11ShaderResourceView* shadowDepthView = nullptr;
        ID3D11UnorderedAccessView* shadowDepthUnorderedView = nullptr;
        ID3D11Buffer* textureDescriptorBuffer = nullptr;
        ID3D11ShaderResourceView* textureDescriptorView = nullptr;
        ID3D11Buffer* textureTexelBuffer = nullptr;
        ID3D11ShaderResourceView* textureTexelView = nullptr;
        ID3D11Buffer* constantBuffer = nullptr;
        ID3D11Buffer* colorBuffer = nullptr;
        ID3D11UnorderedAccessView* colorView = nullptr;
        ID3D11Buffer* depthBuffer = nullptr;
        ID3D11UnorderedAccessView* depthView = nullptr;
        ID3D11Buffer* colorReadback = nullptr;
        ID3D11Buffer* depthReadback = nullptr;
        bool colorsMapped = false;
        bool depthsMapped = false;
        const auto releaseResources = [&]() noexcept
        {
            if (depthsMapped)
            {
                ImmediateContext_->Unmap(depthReadback, 0);
                depthsMapped = false;
            }
            if (colorsMapped)
            {
                ImmediateContext_->Unmap(colorReadback, 0);
                colorsMapped = false;
            }
            ImmediateContext_->ClearState();
            Release(depthReadback);
            Release(colorReadback);
            Release(depthView);
            Release(depthBuffer);
            Release(colorView);
            Release(colorBuffer);
            Release(constantBuffer);
            Release(textureTexelView);
            Release(textureTexelBuffer);
            Release(textureDescriptorView);
            Release(textureDescriptorBuffer);
            Release(shadowDepthUnorderedView);
            Release(shadowDepthView);
            Release(shadowDepthBuffer);
            Release(shadowIndexView);
            Release(shadowIndexBuffer);
            Release(shadowOffsetView);
            Release(shadowOffsetBuffer);
            Release(mainIndexView);
            Release(mainIndexBuffer);
            Release(mainOffsetView);
            Release(mainOffsetBuffer);
            Release(shadowTriangleView);
            Release(shadowTriangleBuffer);
            Release(triangleView);
            Release(triangleBuffer);
            // Software rendering can leave this context idle indefinitely. Submit
            // deferred D3D11 destruction before retiring the frame's large buffers.
            ImmediateContext_->Flush();
        };

        try
        {
            triangleBuffer = CreateStructuredBuffer(
                std::max<std::size_t>(triangles.size(), 1U),
                sizeof(GpuTriangle),
                triangles.empty() ? nullptr : triangles.data(),
                D3D11_BIND_SHADER_RESOURCE);
            triangleView = CreateShaderView(
                triangleBuffer,
                static_cast<UINT>(std::max<std::size_t>(
                    triangles.size(),
                    1U)));

            if (generateShadowMap)
            {
                shadowTriangleBuffer = CreateStructuredBuffer(
                    std::max<std::size_t>(shadowTriangles.size(), 1U),
                    sizeof(GpuTriangle),
                    shadowTriangles.empty()
                        ? nullptr
                        : shadowTriangles.data(),
                    D3D11_BIND_SHADER_RESOURCE);
                shadowTriangleView = CreateShaderView(
                    shadowTriangleBuffer,
                    static_cast<UINT>(std::max<std::size_t>(
                        shadowTriangles.size(),
                        1U)));
            }

            mainOffsetBuffer = CreateStructuredBuffer(
                mainBins.Offsets.size(),
                sizeof(std::uint32_t),
                mainBins.Offsets.data(),
                D3D11_BIND_SHADER_RESOURCE);
            mainOffsetView = CreateShaderView(
                mainOffsetBuffer,
                static_cast<UINT>(mainBins.Offsets.size()));
            mainIndexBuffer = CreateStructuredBuffer(
                std::max<std::size_t>(
                    mainBins.TriangleIndices.size(),
                    1U),
                sizeof(std::uint32_t),
                mainBins.TriangleIndices.empty()
                    ? nullptr
                    : mainBins.TriangleIndices.data(),
                D3D11_BIND_SHADER_RESOURCE);
            mainIndexView = CreateShaderView(
                mainIndexBuffer,
                static_cast<UINT>(std::max<std::size_t>(
                    mainBins.TriangleIndices.size(),
                    1U)));

            if (generateShadowMap)
            {
                shadowOffsetBuffer = CreateStructuredBuffer(
                    shadowBins.Offsets.size(),
                    sizeof(std::uint32_t),
                    shadowBins.Offsets.data(),
                    D3D11_BIND_SHADER_RESOURCE);
                shadowOffsetView = CreateShaderView(
                    shadowOffsetBuffer,
                    static_cast<UINT>(shadowBins.Offsets.size()));
                shadowIndexBuffer = CreateStructuredBuffer(
                    std::max<std::size_t>(
                        shadowBins.TriangleIndices.size(),
                        1U),
                    sizeof(std::uint32_t),
                    shadowBins.TriangleIndices.empty()
                        ? nullptr
                        : shadowBins.TriangleIndices.data(),
                    D3D11_BIND_SHADER_RESOURCE);
                shadowIndexView = CreateShaderView(
                    shadowIndexBuffer,
                    static_cast<UINT>(std::max<std::size_t>(
                        shadowBins.TriangleIndices.size(),
                        1U)));
                shadowDepthBuffer = CreateStructuredBuffer(
                    shadowPixelCount,
                    sizeof(float),
                    nullptr,
                    D3D11_BIND_SHADER_RESOURCE |
                        D3D11_BIND_UNORDERED_ACCESS);
                shadowDepthUnorderedView = CreateUnorderedView(
                    shadowDepthBuffer,
                    static_cast<UINT>(shadowPixelCount));
            }
            else
            {
                const float emptyShadow =
                    std::numeric_limits<float>::infinity();
                const void* shadowData = hasExternalShadowMap
                                             ? static_cast<const void*>(
                                                   settings.SceneShadowMap
                                                       ->Depth.data())
                                             : static_cast<const void*>(
                                                   &emptyShadow);
                shadowDepthBuffer = CreateStructuredBuffer(
                    shadowPixelCount,
                    sizeof(float),
                    shadowData,
                    D3D11_BIND_SHADER_RESOURCE);
            }
            shadowDepthView = CreateShaderView(
                shadowDepthBuffer,
                static_cast<UINT>(shadowPixelCount));

            textureDescriptorBuffer = CreateStructuredBuffer(
                std::max<std::size_t>(textures.Descriptors.size(), 1U),
                sizeof(Detail::GpuRasterTexture),
                textures.Descriptors.empty()
                    ? nullptr
                    : textures.Descriptors.data(),
                D3D11_BIND_SHADER_RESOURCE);
            textureDescriptorView = CreateShaderView(
                textureDescriptorBuffer,
                static_cast<UINT>(std::max<std::size_t>(
                    textures.Descriptors.size(),
                    1U)));
            textureTexelBuffer = CreateStructuredBuffer(
                std::max<std::size_t>(textures.TexelsRGBA8.size(), 1U),
                sizeof(std::uint32_t),
                textures.TexelsRGBA8.empty()
                    ? nullptr
                    : textures.TexelsRGBA8.data(),
                D3D11_BIND_SHADER_RESOURCE);
            textureTexelView = CreateShaderView(
                textureTexelBuffer,
                static_cast<UINT>(std::max<std::size_t>(
                    textures.TexelsRGBA8.size(),
                    1U)));

            D3D11_BUFFER_DESC constantDescription{};
            constantDescription.ByteWidth =
                static_cast<UINT>((sizeof(constants) + 15U) & ~15U);
            constantDescription.Usage = D3D11_USAGE_IMMUTABLE;
            constantDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            D3D11_SUBRESOURCE_DATA constantData{&constants, 0, 0};
            Check(
                Device_->CreateBuffer(
                    &constantDescription,
                    &constantData,
                    &constantBuffer),
                "CreateBuffer(constants)");

            colorBuffer = CreateStructuredBuffer(
                pixelCount,
                sizeof(float) * 4U,
                nullptr,
                D3D11_BIND_UNORDERED_ACCESS);
            depthBuffer = CreateStructuredBuffer(
                pixelCount,
                sizeof(float),
                nullptr,
                D3D11_BIND_UNORDERED_ACCESS);
            colorView = CreateUnorderedView(
                colorBuffer,
                static_cast<UINT>(pixelCount));
            depthView = CreateUnorderedView(
                depthBuffer,
                static_cast<UINT>(pixelCount));
            colorReadback = CreateReadbackBuffer(colorBuffer);
            depthReadback = CreateReadbackBuffer(depthBuffer);

            const auto rasterBegin = std::chrono::steady_clock::now();
            ImmediateContext_->CSSetConstantBuffers(0, 1, &constantBuffer);
            if (generateShadowMap)
            {
                ImmediateContext_->CSSetShader(
                    ShadowShader_,
                    nullptr,
                    0);
                ID3D11ShaderResourceView* shadowResources[] = {
                    shadowTriangleView,
                    shadowOffsetView,
                    shadowIndexView,
                    nullptr,
                    textureDescriptorView,
                    textureTexelView};
                ImmediateContext_->CSSetShaderResources(
                    0,
                    6,
                    shadowResources);
                ID3D11UnorderedAccessView* shadowUnorderedViews[] = {
                    nullptr,
                    nullptr,
                    shadowDepthUnorderedView};
                ImmediateContext_->CSSetUnorderedAccessViews(
                    0,
                    3,
                    shadowUnorderedViews,
                    nullptr);
                ImmediateContext_->Dispatch(
                    (shadowWidth + 7U) / 8U,
                    (shadowHeight + 7U) / 8U,
                    1);
                ID3D11UnorderedAccessView* nullShadowUnordered[] = {
                    nullptr,
                    nullptr,
                    nullptr};
                ID3D11ShaderResourceView* nullShadowResources[] = {
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr};
                ImmediateContext_->CSSetUnorderedAccessViews(
                    0,
                    3,
                    nullShadowUnordered,
                    nullptr);
                ImmediateContext_->CSSetShaderResources(
                    0,
                    6,
                    nullShadowResources);
            }

            if (IsCancelled(externalCancellation, backendCancellation))
            {
                ID3D11Buffer* nullBuffer = nullptr;
                ImmediateContext_->CSSetConstantBuffers(
                    0,
                    1,
                    &nullBuffer);
                ImmediateContext_->CSSetShader(nullptr, nullptr, 0);
                releaseResources();
                return RasterStatus::Cancelled;
            }

            ImmediateContext_->CSSetShader(RasterShader_, nullptr, 0);
            ID3D11ShaderResourceView* rasterResources[] = {
                triangleView,
                mainOffsetView,
                mainIndexView,
                shadowDepthView,
                textureDescriptorView,
                textureTexelView};
            ImmediateContext_->CSSetShaderResources(
                0,
                6,
                rasterResources);
            ID3D11UnorderedAccessView* rasterUnorderedViews[] = {
                colorView,
                depthView};
            ImmediateContext_->CSSetUnorderedAccessViews(
                0,
                2,
                rasterUnorderedViews,
                nullptr);
            const UINT groupsX = (target.Width() + 7U) / 8U;
            const UINT groupsY = (target.Height() + 7U) / 8U;
            ImmediateContext_->Dispatch(groupsX, groupsY, 1);

            ID3D11UnorderedAccessView* nullUnordered[] = {
                nullptr,
                nullptr,
                nullptr};
            ID3D11ShaderResourceView* nullResources[] = {
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr};
            ID3D11Buffer* nullBuffer = nullptr;
            ImmediateContext_->CSSetUnorderedAccessViews(
                0,
                3,
                nullUnordered,
                nullptr);
            ImmediateContext_->CSSetShaderResources(
                0,
                6,
                nullResources);
            ImmediateContext_->CSSetConstantBuffers(
                0,
                1,
                &nullBuffer);
            ImmediateContext_->CSSetShader(nullptr, nullptr, 0);

            ImmediateContext_->CopyResource(colorReadback, colorBuffer);
            ImmediateContext_->CopyResource(depthReadback, depthBuffer);
            D3D11_MAPPED_SUBRESOURCE mappedColors{};
            D3D11_MAPPED_SUBRESOURCE mappedDepths{};
            Check(
                ImmediateContext_->Map(
                    colorReadback,
                    0,
                    D3D11_MAP_READ,
                    0,
                    &mappedColors),
                "Map(colors)");
            colorsMapped = true;
            Check(
                ImmediateContext_->Map(
                    depthReadback,
                    0,
                    D3D11_MAP_READ,
                    0,
                    &mappedDepths),
                "Map(depths)");
            depthsMapped = true;

            const float* colors =
                static_cast<const float*>(mappedColors.pData);
            const float* depths =
                static_cast<const float*>(mappedDepths.pData);
            std::uint64_t shadedPixels = 0;
            for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
            {
                target.MutableColors()[pixel] = {
                    colors[pixel * 4U],
                    colors[pixel * 4U + 1U],
                    colors[pixel * 4U + 2U],
                    colors[pixel * 4U + 3U]};
                target.MutableDepths()[pixel] = depths[pixel];
                if (std::isfinite(depths[pixel]) &&
                    depths[pixel] >= 0.0f &&
                    depths[pixel] <= 1.0f)
                {
                    ++shadedPixels;
                }
            }
            ImmediateContext_->Unmap(depthReadback, 0);
            depthsMapped = false;
            ImmediateContext_->Unmap(colorReadback, 0);
            colorsMapped = false;

            Detail::ApplyGpuSelectionOutline(mainScene, gpuSettings, target,
                externalCancellation, backendCancellation);

            statistics.InputTriangles = scene.Triangles.size();
            statistics.PostClipTriangles = mainScene.Triangles.size();
            statistics.ShadedFragments = shadedPixels;
            statistics.DispatchedTiles =
                static_cast<std::uint64_t>(mainBins.TilesX) *
                    mainBins.TilesY +
                static_cast<std::uint64_t>(shadowBins.TilesX) *
                    shadowBins.TilesY;
            statistics.WorkerThreads = 64;
            statistics.ExecutionPath =
                "Native D3D11 CS 5.0, 16x16 bins=" +
                std::to_string(mainBins.TilesX * mainBins.TilesY) +
                ", refs=" +
                std::to_string(mainBins.TriangleIndices.size()) +
                ", max-list=" +
                std::to_string(mainBins.MaximumListLength) +
                (generateShadowMap
                     ? ", GPU shadow-depth pass"
                     : hasExternalShadowMap
                           ? ", uploaded shadow map"
                           : "");
            statistics.RasterMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - rasterBegin)
                    .count();
        }
        catch (...)
        {
            ID3D11UnorderedAccessView* nullUnordered[] = {
                nullptr,
                nullptr,
                nullptr};
            ID3D11ShaderResourceView* nullResources[] = {
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr};
            ID3D11Buffer* nullBuffer = nullptr;
            ImmediateContext_->CSSetUnorderedAccessViews(
                0,
                3,
                nullUnordered,
                nullptr);
            ImmediateContext_->CSSetShaderResources(
                0,
                6,
                nullResources);
            ImmediateContext_->CSSetConstantBuffers(
                0,
                1,
                &nullBuffer);
            ImmediateContext_->CSSetShader(nullptr, nullptr, 0);
            releaseResources();
            throw;
        }
        releaseResources();
        return IsCancelled(externalCancellation, backendCancellation)
                   ? RasterStatus::Cancelled
                   : RasterStatus::Completed;
    }

private:
    using CreateDeviceFunction = HRESULT(WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT,
        const D3D_FEATURE_LEVEL*, UINT, UINT, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);
    using CompileFunction = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*,
        LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

    static void Check(HRESULT result, const char* operation)
    {
        if (FAILED(result))
        {
            throw std::runtime_error(
                std::string(operation) + " failed with HRESULT " +
                std::to_string(static_cast<unsigned long>(result)));
        }
    }

    [[nodiscard]] ID3D11Buffer* CreateStructuredBuffer(
        std::size_t elementCount,
        std::size_t stride,
        const void* initialData,
        UINT bindFlags)
    {
        if (elementCount == 0 || stride == 0 ||
            elementCount >
                static_cast<std::size_t>(std::numeric_limits<UINT>::max()) / stride)
        {
            throw std::overflow_error("D3D11 structured buffer size is invalid");
        }
        D3D11_BUFFER_DESC description{};
        description.ByteWidth = static_cast<UINT>(elementCount * stride);
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = bindFlags;
        description.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        description.StructureByteStride = static_cast<UINT>(stride);
        D3D11_SUBRESOURCE_DATA data{initialData, 0, 0};
        ID3D11Buffer* buffer = nullptr;
        Check(Device_->CreateBuffer(&description, initialData == nullptr ? nullptr : &data, &buffer), "CreateStructuredBuffer");
        return buffer;
    }

    [[nodiscard]] ID3D11ShaderResourceView* CreateShaderView(
        ID3D11Buffer* buffer,
        UINT elementCount)
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC description{};
        description.Format = DXGI_FORMAT_UNKNOWN;
        description.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        description.Buffer.NumElements = elementCount;
        ID3D11ShaderResourceView* view = nullptr;
        Check(
            Device_->CreateShaderResourceView(buffer, &description, &view),
            "CreateShaderResourceView");
        return view;
    }

    [[nodiscard]] ID3D11UnorderedAccessView* CreateUnorderedView(
        ID3D11Buffer* buffer,
        UINT elementCount)
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC description{};
        description.Format = DXGI_FORMAT_UNKNOWN;
        description.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        description.Buffer.NumElements = elementCount;
        ID3D11UnorderedAccessView* view = nullptr;
        Check(Device_->CreateUnorderedAccessView(buffer, &description, &view), "CreateUnorderedAccessView");
        return view;
    }

    [[nodiscard]] ID3D11Buffer* CreateReadbackBuffer(ID3D11Buffer* source)
    {
        D3D11_BUFFER_DESC description{};
        source->GetDesc(&description);
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = 0;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        description.MiscFlags = 0;
        description.StructureByteStride = 0;
        ID3D11Buffer* buffer = nullptr;
        Check(Device_->CreateBuffer(&description, nullptr, &buffer), "CreateReadbackBuffer");
        return buffer;
    }

    HMODULE D3D11Library_ = nullptr;
    HMODULE CompilerLibrary_ = nullptr;
    CreateDeviceFunction CreateDevice_ = nullptr;
    CompileFunction Compile_ = nullptr;
    ID3D11Device* Device_ = nullptr;
    ID3D11DeviceContext* ImmediateContext_ = nullptr;
    ID3D11ComputeShader* RasterShader_ = nullptr;
    ID3D11ComputeShader* ShadowShader_ = nullptr;
};

#else

class D3D11ComputeRuntime
{
public:
    [[nodiscard]] bool Initialize(std::string& error)
    {
        error = "D3D11 compute raster backend is only available on Windows";
        return false;
    }
    void Shutdown() noexcept {}
};

#endif

} // namespace

class ComputeShaderRasterRHI::Context
{
public:
    Detail::ExecutionGate Execution;
    Detail::StatisticsStore Statistics;
    D3D11ComputeRuntime Runtime;
    mutable std::mutex StateMutex;
    std::string Error;
    bool NativeAvailable = false;
};

ComputeShaderRasterRHI::ComputeShaderRasterRHI()
    : Context_(std::make_unique<Context>())
{
    (void)Initialize({});
}

ComputeShaderRasterRHI::~ComputeShaderRasterRHI()
{
    Shutdown();
}

bool ComputeShaderRasterRHI::Initialize(const RasterRHICreateInfo& createInfo)
{
    Shutdown();
    std::string error;
    const bool available = Context_->Runtime.Initialize(error);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        Context_->NativeAvailable = available;
        Context_->Error = std::move(error);
    }
    return available && Context_->Execution.Initialize(createInfo.WorkerThreads);
}

void ComputeShaderRasterRHI::Shutdown() noexcept
{
    Context_->Execution.Shutdown();
    Context_->Runtime.Shutdown();
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    Context_->NativeAvailable = false;
}

bool ComputeShaderRasterRHI::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return Context_->NativeAvailable && Context_->Execution.IsInitialized();
}

RasterBackend ComputeShaderRasterRHI::Backend() const noexcept
{
    return RasterBackend::ComputeShader;
}

std::string_view ComputeShaderRasterRHI::Name() const noexcept
{
    return "D3D11 Compute Raster";
}

RasterBackendCapabilities ComputeShaderRasterRHI::Capabilities() const
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return {
        RasterBackend::ComputeShader,
        Context_->NativeAvailable,
        Context_->NativeAvailable,
        8,
        8,
        Context_->NativeAvailable
            ? "Native Direct3D 11 CS 5.0 tiled raster and shadow kernels"
            : "Unsupported: " + Context_->Error};
}

RasterStatus ComputeShaderRasterRHI::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    Detail::ValidateRenderTarget(target);
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        if (!Context_->NativeAvailable)
        {
            throw std::runtime_error(
                "D3D11 compute raster backend is unsupported: " + Context_->Error);
        }
    }
    RasterStatistics statistics;
    const RasterStatus status = Context_->Execution.Execute(
        [&](const std::atomic<bool>& backendCancellation)
        {
#if defined(_WIN32)
            return Context_->Runtime.Render(
                scene, settings, target, cancellation, backendCancellation, statistics);
#else
            (void)scene; (void)settings; (void)target; (void)cancellation; (void)backendCancellation;
            return RasterStatus::Cancelled;
#endif
        });
    Context_->Statistics.Set(std::move(statistics));
    return status;
}

void ComputeShaderRasterRHI::Cancel() noexcept
{
    Context_->Execution.Cancel();
}

void ComputeShaderRasterRHI::WaitIdle() noexcept
{
    Context_->Execution.WaitIdle();
}

RasterStatistics ComputeShaderRasterRHI::LastStatistics() const
{
    return Context_->Statistics.Get();
}

} // namespace Evans::Rasterizer
