#include "Rasterizer/MetalRasterRHI.h"

#include "Rasterizer/BackendCommon.hpp"
#include "Rasterizer/GpuRasterCommon.hpp"

#if !defined(__APPLE__)
#error MetalRasterRHI.mm must only be compiled on Apple platforms
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace Evans::Rasterizer
{
namespace
{

constexpr const char* MetalRasterShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RasterVertexGpu
{
    float4 Clip;
    float4 Color;
    float4 Position;
    float4 Normal;
    float4 ShadowClip;
    float4 TexCoord;
};

struct RasterTriangleGpu
{
    RasterVertexGpu Vertices[3];
    float4 BaseColor;
    float4 SpecularColor;
    float4 EmissiveColor;
    float Shininess;
    float AlphaCutoff;
    uint DiffuseTextureIndex;
    uint OpacityTextureIndex;
    uint Selected;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

struct RasterTextureGpu
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct RasterLightGpu
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

struct RasterConstants
{
    float4 World[4];
    float4 View[4];
    float4 Projection[4];
    float4 LightViewProjection[4];
    uint Width;
    uint Height;
    uint TriangleCount;
    uint InputSpace;
    uint DepthRange;
    uint FillModeValue;
    uint ShadingValue;
    uint BackFaceCulling;
    uint FrontFaceCounterClockwise;
    uint OutlineEnabled;
    uint ShadowsEnabled;
    uint ShadowWidth;
    uint ShadowHeight;
    uint TilesX;
    float WireframeWidth;
    float OutlineWidth;
    float ShadowBias;
    float ShadowStrength;
    uint DepthTest;
    uint ShadowPCFRadius;
    float4 ClearColor;
    float4 WireframeColor;
    float4 OutlineColor;
    float4 CameraPosition;
    float4 LightDirectionIntensity;
    float4 LightColorAmbientStrength;
    float4 AmbientColor;
    uint LightCount;
    int ShadowLightIndex;
    uint TextureCount;
    uint TextureTexelCount;
    RasterLightGpu Lights[4];
    float4 ShadowRasterBias;
};

float4 TransformRows(constant float4* rows, float4 value)
{
    return float4(
        dot(rows[0], value),
        dot(rows[1], value),
        dot(rows[2], value),
        dot(rows[3], value));
}

float3 TransformDirection(constant float4* rows, float3 value)
{
    return float3(
        dot(rows[0].xyz, value),
        dot(rows[1].xyz, value),
        dot(rows[2].xyz, value));
}

float Edge(float2 a, float2 b, float2 p)
{
    return (b.x - a.x) * (p.y - a.y) -
           (b.y - a.y) * (p.x - a.x);
}

float3 NormalizeSafe(float3 value)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared < 1.0e-12f
        ? float3(0.0f, 0.0f, 1.0f)
        : value * rsqrt(lengthSquared);
}

float3 NormalizeOr(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared < 1.0e-12f
        ? fallback
        : value * rsqrt(lengthSquared);
}

float SRGBToLinearChannel(float value)
{
    value = saturate(value);
    return value <= 0.04045f
        ? value / 12.92f
        : pow((value + 0.055f) / 1.055f, 2.4f);
}

int WrapTextureCoordinate(int value, uint size)
{
    int result = value % int(size);
    return result < 0 ? result + int(size) : result;
}

float4 ReadTextureTexel(
    RasterTextureGpu descriptor,
    int x,
    int y,
    bool decodeSRGB,
    device const uint* textureTexels)
{
    x = WrapTextureCoordinate(x, descriptor.Width);
    y = WrapTextureCoordinate(y, descriptor.Height);
    const uint packed = textureTexels[
        descriptor.OffsetTexels + uint(y) * descriptor.Width + uint(x)];
    float4 value = float4(uint4(
        packed & 255u,
        (packed >> 8u) & 255u,
        (packed >> 16u) & 255u,
        (packed >> 24u) & 255u)) / 255.0f;
    if (decodeSRGB)
    {
        value.rgb = float3(
            SRGBToLinearChannel(value.r),
            SRGBToLinearChannel(value.g),
            SRGBToLinearChannel(value.b));
    }
    return value;
}

float4 SampleTexture(
    uint textureIndex,
    float2 texCoord,
    bool decodeSRGB,
    constant RasterConstants& constants,
    device const RasterTextureGpu* textureDescriptors,
    device const uint* textureTexels)
{
    if (textureIndex >= constants.TextureCount)
    {
        return float4(1.0f);
    }
    const RasterTextureGpu descriptor = textureDescriptors[textureIndex];
    if (descriptor.Width == 0u ||
        descriptor.Height == 0u ||
        descriptor.OffsetTexels >= constants.TextureTexelCount)
    {
        return float4(1.0f);
    }
    const uint availableTexels =
        constants.TextureTexelCount - descriptor.OffsetTexels;
    if (descriptor.Height > availableTexels / descriptor.Width)
    {
        return float4(1.0f);
    }

    const float x = fract(texCoord.x) * float(descriptor.Width) - 0.5f;
    const float y =
        fract(1.0f - texCoord.y) * float(descriptor.Height) - 0.5f;
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const float blendX = x - floor(x);
    const float blendY = y - floor(y);
    const float4 top = mix(
        ReadTextureTexel(
            descriptor,
            x0,
            y0,
            decodeSRGB,
            textureTexels),
        ReadTextureTexel(
            descriptor,
            x0 + 1,
            y0,
            decodeSRGB,
            textureTexels),
        blendX);
    const float4 bottom = mix(
        ReadTextureTexel(
            descriptor,
            x0,
            y0 + 1,
            decodeSRGB,
            textureTexels),
        ReadTextureTexel(
            descriptor,
            x0 + 1,
            y0 + 1,
            decodeSRGB,
            textureTexels),
        blendX);
    return mix(top, bottom, blendY);
}

float4 SampleSurface(
    device const RasterTriangleGpu& primitive,
    float p0,
    float p1,
    float p2,
    float denominator,
    constant RasterConstants& constants,
    device const RasterTextureGpu* textureDescriptors,
    device const uint* textureTexels)
{
    const float4 vertex =
        (primitive.Vertices[0].Color * p0 +
         primitive.Vertices[1].Color * p1 +
         primitive.Vertices[2].Color * p2) /
        denominator;
    const float2 texCoord =
        (primitive.Vertices[0].TexCoord.xy * p0 +
         primitive.Vertices[1].TexCoord.xy * p1 +
         primitive.Vertices[2].TexCoord.xy * p2) /
        denominator;
    const float4 diffuse = SampleTexture(
        primitive.DiffuseTextureIndex,
        texCoord,
        true,
        constants,
        textureDescriptors,
        textureTexels);
    const float opacity = SampleTexture(
        primitive.OpacityTextureIndex,
        texCoord,
        false,
        constants,
        textureDescriptors,
        textureTexels).r;
    return float4(
        vertex.rgb * primitive.BaseColor.rgb * diffuse.rgb,
        saturate(
            vertex.a * primitive.BaseColor.a * diffuse.a * opacity));
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    float shininess,
    uint shading,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    const float directionLengthSquared = dot(lightDirection, lightDirection);
    if (directionLengthSquared <= 1.0e-12f)
    {
        return;
    }
    const float3 direction = lightDirection * rsqrt(directionLengthSquared);
    const float normalToLight = max(dot(normal, direction), 0.0f);
    if (normalToLight <= 0.0f)
    {
        return;
    }
    const float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    if (shading == 2u)
    {
        const float3 halfVector = NormalizeOr(direction + viewDirection, normal);
        const float specularAmount = pow(
            max(dot(normal, halfVector), 0.0f),
            max(shininess, 1.0f));
        specularLight += visibleRadiance * specularAmount;
    }
}

void AccumulateSceneLight(
    uint lightIndex,
    float3 position,
    float3 normal,
    float3 viewDirection,
    float shininess,
    uint shading,
    float shadowVisibility,
    constant RasterConstants& constants,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    const RasterLightGpu light = constants.Lights[lightIndex];
    const uint type = uint(light.PositionType.w + 0.5f);
    const float visibility =
        int(lightIndex) == constants.ShadowLightIndex &&
                light.ColorCastShadows.w > 0.5f
            ? shadowVisibility
            : 1.0f;
    const float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0f) *
        max(light.DirectionIntensity.w, 0.0f);
    if (type == 0u)
    {
        AccumulateLightSample(
            light.DirectionIntensity.xyz,
            radiance,
            visibility,
            normal,
            viewDirection,
            shininess,
            shading,
            diffuseLight,
            specularLight);
        return;
    }
    if (type == 1u)
    {
        const float3 toLight = light.PositionType.xyz - position;
        const float distanceSquared = max(dot(toLight, toLight), 1.0e-6f);
        AccumulateLightSample(
            toLight,
            radiance / distanceSquared,
            visibility,
            normal,
            viewDirection,
            shininess,
            shading,
            diffuseLight,
            specularLight);
        return;
    }
    const float3 emissionDirection = NormalizeOr(
        light.DirectionIntensity.xyz,
        float3(0.0f, -1.0f, 0.0f));
    const float3 helper = abs(emissionDirection.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = NormalizeOr(
        cross(helper, emissionDirection),
        float3(1.0f, 0.0f, 0.0f));
    const float3 bitangent = NormalizeOr(
        cross(emissionDirection, tangent),
        float3(0.0f, 0.0f, 1.0f));
    for (int sampleY = 0; sampleY < 2; ++sampleY)
    {
        for (int sampleX = 0; sampleX < 2; ++sampleX)
        {
            const float offsetX = sampleX == 0 ? -0.25f : 0.25f;
            const float offsetY = sampleY == 0 ? -0.25f : 0.25f;
            const float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            const float3 toLight = samplePosition - position;
            const float distanceSquared = max(dot(toLight, toLight), 1.0e-6f);
            const float3 direction = toLight * rsqrt(distanceSquared);
            const float emitterCosine = max(
                dot(emissionDirection, -direction),
                0.0f);
            AccumulateLightSample(
                direction,
                radiance * (0.25f * emitterCosine / distanceSquared),
                visibility,
                normal,
                viewDirection,
                shininess,
                shading,
                diffuseLight,
                specularLight);
        }
    }
}

float4 LightClip(
    device const RasterVertexGpu& vertex,
    constant RasterConstants& constants)
{
    return constants.InputSpace == 0
        ? vertex.ShadowClip
        : TransformRows(
              constants.LightViewProjection,
              TransformRows(constants.World, vertex.Position));
}

kernel void EvansMetalShadowMain(
    device const RasterTriangleGpu* triangles [[buffer(0)]],
    device const uint* tileOffsets [[buffer(1)]],
    device const uint* tileTriangleIndices [[buffer(2)]],
    constant RasterConstants& constants [[buffer(3)]],
    device float* generatedShadowDepths [[buffer(4)]],
    constant uint& dispatchOriginY [[buffer(5)]],
    device const RasterTextureGpu* textureDescriptors [[buffer(6)]],
    device const uint* textureTexels [[buffer(7)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint x = id.x;
    const uint y = id.y + dispatchOriginY;
    if (x >= constants.ShadowWidth || y >= constants.ShadowHeight)
    {
        return;
    }

    const uint pixel = y * constants.ShadowWidth + x;
    const uint shadowTilesX = (constants.ShadowWidth + 15u) / 16u;
    const uint tile = (y / 16u) * shadowTilesX + x / 16u;
    const float2 samplePosition = float2(float(x) + 0.5f, float(y) + 0.5f);
    float bestDepth = 1.0e30f;
    for (uint binIndex = tileOffsets[tile];
         binIndex < tileOffsets[tile + 1u];
         ++binIndex)
    {
        device const RasterTriangleGpu& primitive =
            triangles[tileTriangleIndices[binIndex]];
        float2 screen[3];
        float inverseW[3];
        float vertexDepth[3];
        bool valid = true;
        for (uint vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            const float4 clip =
                LightClip(primitive.Vertices[vertexIndex], constants);
            if (clip.w <= 1.0e-6f)
            {
                valid = false;
                break;
            }
            inverseW[vertexIndex] = 1.0f / clip.w;
            const float3 ndc = clip.xyz * inverseW[vertexIndex];
            screen[vertexIndex] = float2(
                (ndc.x * 0.5f + 0.5f) * float(constants.ShadowWidth),
                (0.5f - ndc.y * 0.5f) * float(constants.ShadowHeight));
            vertexDepth[vertexIndex] = constants.DepthRange == 0
                ? ndc.z
                : ndc.z * 0.5f + 0.5f;
        }
        if (!valid)
        {
            continue;
        }

        const float area = Edge(screen[0], screen[1], screen[2]);
        if (abs(area) < 1.0e-6f)
        {
            continue;
        }
        const float w0 =
            Edge(screen[1], screen[2], samplePosition) / area;
        const float w1 =
            Edge(screen[2], screen[0], samplePosition) / area;
        const float w2 = 1.0f - w0 - w1;
        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
        {
            continue;
        }
        float depth =
            w0 * vertexDepth[0] +
            w1 * vertexDepth[1] +
            w2 * vertexDepth[2];
        if (depth < 0.0f || depth > 1.0f) continue;
        if (constants.ShadowRasterBias.x > 0.0f)
        {
            const float dz1 = vertexDepth[1] - vertexDepth[0];
            const float dz2 = vertexDepth[2] - vertexDepth[0];
            const float slopeX = (dz1 * (screen[2].y - screen[0].y) -
                dz2 * (screen[1].y - screen[0].y)) / area;
            const float slopeY = ((screen[1].x - screen[0].x) * dz2 -
                (screen[2].x - screen[0].x) * dz1) / area;
            depth = min(depth + constants.ShadowRasterBias.x * max(abs(slopeX), abs(slopeY)), 1.0f);
        }
        if (depth >= bestDepth)
        {
            continue;
        }
        const float p0 = w0 * inverseW[0];
        const float p1 = w1 * inverseW[1];
        const float p2 = w2 * inverseW[2];
        const float denominator = p0 + p1 + p2;
        if (abs(denominator) < 1.0e-8f)
        {
            continue;
        }
        const float alpha = SampleSurface(
            primitive,
            p0,
            p1,
            p2,
            denominator,
            constants,
            textureDescriptors,
            textureTexels).a;
        if (primitive.AlphaCutoff > 0.0f &&
            alpha < primitive.AlphaCutoff)
        {
            continue;
        }
        bestDepth = depth;
    }
    generatedShadowDepths[pixel] = bestDepth;
}

kernel void EvansMetalRasterMain(
    device const RasterTriangleGpu* triangles [[buffer(0)]],
    device const uint* tileOffsets [[buffer(1)]],
    device const uint* tileTriangleIndices [[buffer(2)]],
    device const float* shadowDepths [[buffer(3)]],
    constant RasterConstants& constants [[buffer(4)]],
    device float4* outputColors [[buffer(5)]],
    device float* outputDepths [[buffer(6)]],
    constant uint& dispatchOriginY [[buffer(7)]],
    device const RasterTextureGpu* textureDescriptors [[buffer(8)]],
    device const uint* textureTexels [[buffer(9)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint x = id.x;
    const uint y = id.y + dispatchOriginY;
    if (x >= constants.Width || y >= constants.Height)
    {
        return;
    }

    const uint pixel = y * constants.Width + x;
    const uint tile = (y / 16u) * constants.TilesX + x / 16u;
    const float2 samplePosition = float2(float(x) + 0.5f, float(y) + 0.5f);
    float bestDepth = 1.0e30f;
    float4 bestColor = constants.ClearColor;
    float bestWireDepth = 1.0e30f;
    for (uint binIndex = tileOffsets[tile];
         binIndex < tileOffsets[tile + 1u];
         ++binIndex)
    {
        device const RasterTriangleGpu& primitive =
            triangles[tileTriangleIndices[binIndex]];
        float4 clip[3];
        float4 world[3];
        float4 shadowClip[3];
        float3 normal[3];
        float2 screen[3];
        float inverseW[3];
        float vertexDepth[3];
        bool valid = true;
        for (uint vertexIndex = 0; vertexIndex < 3; ++vertexIndex)
        {
            device const RasterVertexGpu& vertex =
                primitive.Vertices[vertexIndex];
            if (constants.InputSpace == 0)
            {
                clip[vertexIndex] = vertex.Clip;
                world[vertexIndex] = vertex.Position;
                normal[vertexIndex] = vertex.Normal.xyz;
                shadowClip[vertexIndex] = vertex.ShadowClip;
            }
            else
            {
                world[vertexIndex] =
                    TransformRows(constants.World, vertex.Position);
                clip[vertexIndex] = TransformRows(
                    constants.Projection,
                    TransformRows(constants.View, world[vertexIndex]));
                normal[vertexIndex] = NormalizeSafe(
                    TransformDirection(
                        constants.World,
                        vertex.Normal.xyz));
                shadowClip[vertexIndex] = TransformRows(
                    constants.LightViewProjection,
                    world[vertexIndex]);
            }
            if (clip[vertexIndex].w <= 1.0e-6f)
            {
                valid = false;
                break;
            }
            inverseW[vertexIndex] = 1.0f / clip[vertexIndex].w;
            const float3 ndc =
                clip[vertexIndex].xyz * inverseW[vertexIndex];
            screen[vertexIndex] = float2(
                (ndc.x * 0.5f + 0.5f) * float(constants.Width),
                (0.5f - ndc.y * 0.5f) * float(constants.Height));
            vertexDepth[vertexIndex] = constants.DepthRange == 0
                ? ndc.z
                : ndc.z * 0.5f + 0.5f;
        }
        if (!valid)
        {
            continue;
        }

        const float area = Edge(screen[0], screen[1], screen[2]);
        if (abs(area) < 1.0e-6f)
        {
            continue;
        }
        const bool front = constants.FrontFaceCounterClockwise != 0
            ? area < 0.0f
            : area > 0.0f;
        if (constants.BackFaceCulling != 0 && !front)
        {
            continue;
        }
        float w0 = Edge(screen[1], screen[2], samplePosition) / area;
        float w1 = Edge(screen[2], screen[0], samplePosition) / area;
        float w2 = 1.0f - w0 - w1;
        const bool inside = w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f;
        bool wire = false;
        float3 edgeWeights(0.0f);
        if (constants.FillModeValue != 0)
        {
            float nearestEdge = 1.0e30f;
            for (int edgeIndex = 0; edgeIndex < 3; ++edgeIndex)
            {
                const int next = (edgeIndex + 1) % 3;
                const float2 segment = screen[next] - screen[edgeIndex];
                const float t = clamp(dot(samplePosition - screen[edgeIndex], segment) /
                    max(dot(segment, segment), 1.0e-12f), 0.0f, 1.0f);
                const float2 offset = samplePosition - (screen[edgeIndex] + t * segment);
                const float distanceSquared = dot(offset, offset);
                if (distanceSquared < nearestEdge)
                {
                    nearestEdge = distanceSquared;
                    edgeWeights = float3(0.0f);
                    edgeWeights[edgeIndex] = 1.0f - t;
                    edgeWeights[next] = t;
                }
            }
            const float halfWidth = max(constants.WireframeWidth, 1.0f) * 0.5f;
            wire = nearestEdge <= halfWidth * halfWidth;
        }
        if (wire)
        {
            const float wireDepth = dot(edgeWeights, float3(vertexDepth[0], vertexDepth[1], vertexDepth[2]));
            const float3 edgePerspective = edgeWeights * float3(inverseW[0], inverseW[1], inverseW[2]);
            const float edgeDenominator = edgePerspective.x + edgePerspective.y + edgePerspective.z;
            if (wireDepth >= 0.0f && wireDepth <= 1.0f && abs(edgeDenominator) > 1.0e-8f)
            {
                const float edgeAlpha = SampleSurface(primitive, edgePerspective.x, edgePerspective.y,
                    edgePerspective.z, edgeDenominator, constants, textureDescriptors, textureTexels).a;
                if (primitive.AlphaCutoff <= 0.0f || edgeAlpha >= primitive.AlphaCutoff)
                    bestWireDepth = min(bestWireDepth, wireDepth);
            }
        }
        if (constants.FillModeValue == 1 || !inside) continue;

        const float depth =
            w0 * vertexDepth[0] +
            w1 * vertexDepth[1] +
            w2 * vertexDepth[2];
        if (depth < 0.0f || depth > 1.0f ||
            (constants.DepthTest != 0 && depth >= bestDepth))
        {
            continue;
        }

        const float p0 = w0 * inverseW[0];
        const float p1 = w1 * inverseW[1];
        const float p2 = w2 * inverseW[2];
        const float denominator = p0 + p1 + p2;
        if (abs(denominator) < 1.0e-8f)
        {
            continue;
        }
        float4 color = SampleSurface(
            primitive,
            p0,
            p1,
            p2,
            denominator,
            constants,
            textureDescriptors,
            textureTexels);
        if (primitive.AlphaCutoff > 0.0f &&
            color.a < primitive.AlphaCutoff)
        {
            continue;
        }

        float shadowVisibility = 1.0f;
        if (constants.ShadowsEnabled != 0 &&
            constants.ShadowWidth > 0 &&
            constants.ShadowHeight > 0)
        {
            const float4 shadow =
                (shadowClip[0] * p0 +
                 shadowClip[1] * p1 +
                 shadowClip[2] * p2) /
                denominator;
            if (shadow.w > 1.0e-6f)
            {
                const float inverseShadowW = 1.0f / shadow.w;
                const float shadowX =
                    shadow.x * inverseShadowW * 0.5f + 0.5f;
                const float shadowY =
                    0.5f - shadow.y * inverseShadowW * 0.5f;
                float shadowZ = shadow.z * inverseShadowW;
                if (constants.DepthRange != 0)
                {
                    shadowZ = shadowZ * 0.5f + 0.5f;
                }
                if (shadowX >= 0.0f && shadowX <= 1.0f &&
                    shadowY >= 0.0f && shadowY <= 1.0f &&
                    shadowZ >= 0.0f && shadowZ <= 1.0f)
                {
                    const int centerX = min(
                        int(shadowX * float(constants.ShadowWidth)),
                        int(constants.ShadowWidth) - 1);
                    const int centerY = min(
                        int(shadowY * float(constants.ShadowHeight)),
                        int(constants.ShadowHeight) - 1);
                    const int radius =
                        int(min(constants.ShadowPCFRadius, 8u));
                    uint occluded = 0;
                    uint validSamples = 0;
                    for (int offsetY = -radius;
                         offsetY <= radius;
                         ++offsetY)
                    {
                        const int sampleY = centerY + offsetY;
                        if (sampleY < 0 ||
                            sampleY >= int(constants.ShadowHeight))
                        {
                            continue;
                        }
                        for (int offsetX = -radius;
                             offsetX <= radius;
                             ++offsetX)
                        {
                            const int sampleX = centerX + offsetX;
                            if (sampleX < 0 ||
                                sampleX >= int(constants.ShadowWidth))
                            {
                                continue;
                            }
                            ++validSamples;
                            const uint shadowIndex =
                                uint(sampleY) * constants.ShadowWidth +
                                uint(sampleX);
                            if (shadowZ - constants.ShadowBias >
                                shadowDepths[shadowIndex])
                            {
                                ++occluded;
                            }
                        }
                    }
                    if (validSamples > 0)
                    {
                        shadowVisibility =
                            1.0f - saturate(constants.ShadowStrength) *
                            (float(occluded) / float(validSamples));
                    }
                }
            }
        }

        if (constants.ShadingValue == 0)
        {
            color.rgb += primitive.EmissiveColor.rgb;
        }
        else
        {
            float3 surfaceNormal = NormalizeSafe(
                (normal[0] * p0 +
                 normal[1] * p1 +
                 normal[2] * p2) /
                denominator);
            if (!front)
            {
                surfaceNormal = -surfaceNormal;
            }
            const float4 interpolatedPosition =
                (world[0] * p0 +
                 world[1] * p1 +
                 world[2] * p2) /
                denominator;
            const float3 position = interpolatedPosition.xyz;
            const float3 viewDirection = NormalizeOr(
                constants.CameraPosition.xyz - position,
                surfaceNormal);
            float3 diffuseLight = float3(0.0f);
            float3 specularLight = float3(0.0f);
            const uint lightCount = min(constants.LightCount, 4u);
            for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
            {
                AccumulateSceneLight(
                    lightIndex,
                    position,
                    surfaceNormal,
                    viewDirection,
                    primitive.Shininess,
                    constants.ShadingValue,
                    shadowVisibility,
                    constants,
                    diffuseLight,
                    specularLight);
            }
            color.rgb =
                color.rgb *
                    (constants.AmbientColor.rgb *
                         constants.LightColorAmbientStrength.w +
                     diffuseLight) +
                primitive.SpecularColor.rgb * specularLight +
                primitive.EmissiveColor.rgb;
        }

        color.rgb = max(color.rgb, 0.0f);
        color.rgb /= max(
            1.0f,
            max(color.r, max(color.g, color.b)));
        bestDepth = depth;
        bestColor = float4(saturate(color.rgb), saturate(color.a));
    }

    if (constants.FillModeValue == 1) bestDepth = bestWireDepth;
    if (constants.FillModeValue != 0 && bestWireDepth <= 1.0f &&
        (constants.DepthTest == 0 || bestWireDepth <= bestDepth + 1.0e-4f))
    {
        bestColor = constants.WireframeColor;
        bestDepth = min(bestDepth, bestWireDepth);
    }
    outputColors[pixel] = bestColor;
    outputDepths[pixel] = bestDepth;
}
)MSL";

template <typename ObjectType>
void ReleaseObject(ObjectType& object) noexcept
{
    if (object != nil)
    {
        [object release];
        object = nil;
    }
}

[[nodiscard]] std::string ErrorDescription(
    NSError* error,
    const char* fallback)
{
    if (error != nil)
    {
        const char* text = [[error localizedDescription] UTF8String];
        if (text != nullptr)
        {
            return text;
        }
    }
    return fallback;
}

[[nodiscard]] std::string DeviceDescription(id<MTLDevice> device)
{
    if (device == nil)
    {
        return {};
    }
    const char* name = [[device name] UTF8String];
    return name != nullptr ? name : "Apple GPU";
}

struct MetalRasterBuffers
{
    ~MetalRasterBuffers()
    {
        ReleaseObject(OutputDepths);
        ReleaseObject(OutputColors);
        ReleaseObject(ShadowDepths);
        ReleaseObject(TextureTexels);
        ReleaseObject(TextureDescriptors);
        ReleaseObject(ShadowTriangleIndices);
        ReleaseObject(ShadowOffsets);
        ReleaseObject(ShadowTriangles);
        ReleaseObject(MainTriangleIndices);
        ReleaseObject(MainOffsets);
        ReleaseObject(MainTriangles);
    }

    id<MTLBuffer> MainTriangles = nil;
    id<MTLBuffer> MainOffsets = nil;
    id<MTLBuffer> MainTriangleIndices = nil;
    id<MTLBuffer> ShadowTriangles = nil;
    id<MTLBuffer> ShadowOffsets = nil;
    id<MTLBuffer> ShadowTriangleIndices = nil;
    id<MTLBuffer> TextureDescriptors = nil;
    id<MTLBuffer> TextureTexels = nil;
    id<MTLBuffer> ShadowDepths = nil;
    id<MTLBuffer> OutputColors = nil;
    id<MTLBuffer> OutputDepths = nil;
};

} // namespace

class MetalRasterRHI::Context
{
public:
    ~Context()
    {
        ShutdownNative();
    }

    [[nodiscard]] bool InitializeNative(std::string& error)
    {
        ShutdownNative();
        Device = MTLCreateSystemDefaultDevice();
        if (Device == nil)
        {
            error = "No Metal device is available";
            return false;
        }
        Queue = [Device newCommandQueue];
        if (Queue == nil)
        {
            error = "Metal failed to create a compute command queue";
            ShutdownNative();
            return false;
        }

        NSError* compileError = nil;
        NSString* source =
            [NSString stringWithUTF8String:MetalRasterShaderSource];
        id<MTLLibrary> library =
            [Device newLibraryWithSource:source
                                options:nil
                                  error:&compileError];
        if (library == nil)
        {
            error = ErrorDescription(
                compileError,
                "Metal raster shader compilation failed");
            ShutdownNative();
            return false;
        }

        id<MTLFunction> shadowFunction =
            [library newFunctionWithName:@"EvansMetalShadowMain"];
        id<MTLFunction> rasterFunction =
            [library newFunctionWithName:@"EvansMetalRasterMain"];
        if (shadowFunction == nil || rasterFunction == nil)
        {
            error = "Metal raster shader entry points are unavailable";
            ReleaseObject(rasterFunction);
            ReleaseObject(shadowFunction);
            ReleaseObject(library);
            ShutdownNative();
            return false;
        }

        NSError* shadowError = nil;
        ShadowPipeline =
            [Device newComputePipelineStateWithFunction:shadowFunction
                                                   error:&shadowError];
        NSError* rasterError = nil;
        RasterPipeline =
            [Device newComputePipelineStateWithFunction:rasterFunction
                                                   error:&rasterError];
        ReleaseObject(rasterFunction);
        ReleaseObject(shadowFunction);
        ReleaseObject(library);
        if (ShadowPipeline == nil || RasterPipeline == nil)
        {
            error = ShadowPipeline == nil
                ? ErrorDescription(
                      shadowError,
                      "Metal shadow pipeline creation failed")
                : ErrorDescription(
                      rasterError,
                      "Metal raster pipeline creation failed");
            ShutdownNative();
            return false;
        }

        DeviceName = DeviceDescription(Device);
        return true;
    }

    void ShutdownNative() noexcept
    {
        ReleaseObject(RasterPipeline);
        ReleaseObject(ShadowPipeline);
        ReleaseObject(Queue);
        ReleaseObject(Device);
        DeviceName.clear();
    }

    void MarkTerminalFailure(std::string error)
    {
        std::lock_guard<std::mutex> lock(StateMutex);
        NativeAvailable = false;
        Error = std::move(error);
    }

    void ThrowIfNativeUnavailable() const
    {
        std::string error;
        {
            std::lock_guard<std::mutex> lock(StateMutex);
            if (NativeAvailable)
            {
                return;
            }
            error = Error;
        }
        throw std::runtime_error(
            "Metal raster backend is unsupported: " + error);
    }

    [[nodiscard]] id<MTLBuffer> CreateBuffer(
        std::size_t count,
        std::size_t stride,
        const void* data,
        const char* label) const
    {
        if (stride == 0 ||
            count > std::numeric_limits<std::size_t>::max() / stride)
        {
            throw std::overflow_error(
                std::string("Metal buffer size overflow: ") + label);
        }
        const std::size_t requestedLength = count * stride;
        const std::size_t length = std::max(requestedLength, stride);
        if (length > std::numeric_limits<NSUInteger>::max())
        {
            throw std::overflow_error(
                std::string("Metal buffer is too large: ") + label);
        }

        id<MTLBuffer> buffer = data != nullptr && requestedLength != 0
            ? [Device
                  newBufferWithBytes:data
                              length:static_cast<NSUInteger>(requestedLength)
                             options:MTLResourceStorageModeShared]
            : [Device
                  newBufferWithLength:static_cast<NSUInteger>(length)
                              options:MTLResourceStorageModeShared];
        if (buffer == nil)
        {
            throw std::runtime_error(
                std::string("Metal failed to allocate buffer: ") + label);
        }
        if (data == nullptr)
        {
            std::memset([buffer contents], 0, length);
        }
        [buffer setLabel:[NSString stringWithUTF8String:label]];
        return buffer;
    }

    [[nodiscard]] bool CommandSucceeded(
        id<MTLCommandBuffer> commandBuffer,
        std::string& error) const
    {
        if ([commandBuffer status] == MTLCommandBufferStatusCompleted)
        {
            return true;
        }
        error = ErrorDescription(
            [commandBuffer error],
            "Metal raster command execution failed");
        return false;
    }

    [[nodiscard]] bool CommitAndWait(
        id<MTLCommandBuffer> commandBuffer,
        std::string& error)
    {
        constexpr auto CommandTimeout = std::chrono::seconds(30);
        [commandBuffer commit];
        const auto deadline =
            std::chrono::steady_clock::now() + CommandTimeout;
        while ([commandBuffer status] != MTLCommandBufferStatusCompleted &&
               [commandBuffer status] != MTLCommandBufferStatusError)
        {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                error =
                    "Terminal Metal raster backend failure: a command timed "
                    "out after 30 seconds; the backend was disabled because "
                    "the submitted command may still be in flight";
                MarkTerminalFailure(error);
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!CommandSucceeded(commandBuffer, error))
        {
            error = "Terminal Metal raster backend failure: " + error;
            MarkTerminalFailure(error);
            return false;
        }
        return true;
    }

    [[nodiscard]] RasterStatus RenderNative(
        const RasterScene& scene,
        const RasterSettings& settings,
        RenderTarget& target,
        const RasterCancellationToken& externalCancellation,
        const std::atomic<bool>& backendCancellation,
        RasterStatistics& statistics)
    {
        if (Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const bool hasExternalShadowMap =
            settings.ShadowsEnabled &&
            settings.SceneShadowMap != nullptr &&
            settings.SceneShadowMap->IsValid();
        const bool generateShadowMap =
            settings.ShadowsEnabled &&
            !hasExternalShadowMap &&
            settings.AutoGenerateShadowMap;
        const bool shadowsEnabled =
            hasExternalShadowMap || generateShadowMap;
        const std::uint32_t shadowWidth = hasExternalShadowMap
            ? settings.SceneShadowMap->Width
            : (generateShadowMap
                   ? std::clamp(settings.ShadowMapSize, 16U, 4096U)
                   : 0U);
        const std::uint32_t shadowHeight = hasExternalShadowMap
            ? settings.SceneShadowMap->Height
            : shadowWidth;

        const auto transformBegin = std::chrono::steady_clock::now();
        Detail::PreparedGpuRasterScenes prepared =
            Detail::PrepareGpuRasterScenes(
                scene,
                settings,
                generateShadowMap,
                externalCancellation,
                backendCancellation);
        if (prepared.Cancelled ||
            Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const RasterScene& mainScene = prepared.Main;
        const RasterScene& shadowScene = prepared.Shadow;
        const RasterSettings& gpuSettings = prepared.Settings;
        const Detail::GpuRasterTileBins mainBins =
            Detail::BuildGpuRasterTileBins(
                mainScene,
                gpuSettings,
                target.Width(),
                target.Height(),
                false,
                externalCancellation,
                backendCancellation);
        if (mainBins.Cancelled ||
            Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }
        const Detail::GpuRasterTileBins shadowBins = generateShadowMap
            ? Detail::BuildGpuRasterTileBins(
                  shadowScene,
                  gpuSettings,
                  shadowWidth,
                  shadowHeight,
                  true,
                  externalCancellation,
                  backendCancellation)
            : Detail::GpuRasterTileBins{};
        if (shadowBins.Cancelled ||
            Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const std::vector<Detail::GpuRasterTriangle> triangles =
            Detail::BuildGpuRasterTriangles(mainScene);
        const std::vector<Detail::GpuRasterTriangle> shadowTriangles =
            generateShadowMap
                ? Detail::BuildGpuRasterTriangles(shadowScene)
                : std::vector<Detail::GpuRasterTriangle>{};
        const Detail::GpuRasterTextureData textures =
            Detail::BuildGpuRasterTextures(mainScene);
        if (Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const Detail::GpuRasterConstants constants =
            Detail::BuildGpuRasterConstants(
                mainScene,
                gpuSettings,
                target,
                shadowsEnabled,
                shadowWidth,
                shadowHeight);
        statistics.TransformMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - transformBegin)
                .count();

        const std::size_t pixelCount =
            static_cast<std::size_t>(target.Width()) * target.Height();
        const std::size_t shadowPixelCount =
            std::max<std::size_t>(
                static_cast<std::size_t>(shadowWidth) * shadowHeight,
                1U);
        MetalRasterBuffers buffers;
        buffers.MainTriangles = CreateBuffer(
            triangles.size(),
            sizeof(Detail::GpuRasterTriangle),
            triangles.empty() ? nullptr : triangles.data(),
            "Evans Metal raster triangles");
        buffers.MainOffsets = CreateBuffer(
            mainBins.Offsets.size(),
            sizeof(std::uint32_t),
            mainBins.Offsets.data(),
            "Evans Metal raster tile offsets");
        buffers.MainTriangleIndices = CreateBuffer(
            mainBins.TriangleIndices.size(),
            sizeof(std::uint32_t),
            mainBins.TriangleIndices.empty()
                ? nullptr
                : mainBins.TriangleIndices.data(),
            "Evans Metal raster tile indices");
        buffers.TextureDescriptors = CreateBuffer(
            textures.Descriptors.size(),
            sizeof(Detail::GpuRasterTexture),
            textures.Descriptors.empty()
                ? nullptr
                : textures.Descriptors.data(),
            "Evans Metal raster texture descriptors");
        buffers.TextureTexels = CreateBuffer(
            textures.TexelsRGBA8.size(),
            sizeof(std::uint32_t),
            textures.TexelsRGBA8.empty()
                ? nullptr
                : textures.TexelsRGBA8.data(),
            "Evans Metal raster texture texels");

        if (generateShadowMap)
        {
            buffers.ShadowTriangles = CreateBuffer(
                shadowTriangles.size(),
                sizeof(Detail::GpuRasterTriangle),
                shadowTriangles.empty()
                    ? nullptr
                    : shadowTriangles.data(),
                "Evans Metal shadow triangles");
            buffers.ShadowOffsets = CreateBuffer(
                shadowBins.Offsets.size(),
                sizeof(std::uint32_t),
                shadowBins.Offsets.data(),
                "Evans Metal shadow tile offsets");
            buffers.ShadowTriangleIndices = CreateBuffer(
                shadowBins.TriangleIndices.size(),
                sizeof(std::uint32_t),
                shadowBins.TriangleIndices.empty()
                    ? nullptr
                    : shadowBins.TriangleIndices.data(),
                "Evans Metal shadow tile indices");
        }

        const float emptyShadow =
            std::numeric_limits<float>::infinity();
        const void* shadowData = hasExternalShadowMap
            ? static_cast<const void*>(
                  settings.SceneShadowMap->Depth.data())
            : (generateShadowMap
                   ? nullptr
                   : static_cast<const void*>(&emptyShadow));
        buffers.ShadowDepths = CreateBuffer(
            shadowPixelCount,
            sizeof(float),
            shadowData,
            "Evans Metal shadow depths");
        buffers.OutputColors = CreateBuffer(
            pixelCount,
            sizeof(float) * 4U,
            nullptr,
            "Evans Metal raster colors");
        buffers.OutputDepths = CreateBuffer(
            pixelCount,
            sizeof(float),
            nullptr,
            "Evans Metal raster depths");

        if (Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const auto rasterBegin = std::chrono::steady_clock::now();
        constexpr std::uint32_t RowsPerSubmission = 32U;
        const MTLSize threadsPerGroup = MTLSizeMake(8, 8, 1);
        if (generateShadowMap)
        {
            for (std::uint32_t row = 0;
                 row < shadowHeight;
                 row += RowsPerSubmission)
            {
                if (Detail::IsGpuRasterCancelled(
                        externalCancellation,
                        backendCancellation))
                {
                    return RasterStatus::Cancelled;
                }
                @autoreleasepool
                {
                    id<MTLCommandBuffer> commandBuffer =
                        [Queue commandBuffer];
                    if (commandBuffer == nil)
                    {
                        throw std::runtime_error(
                            "Metal failed to create a shadow command buffer");
                    }
                    [commandBuffer
                        setLabel:@"Evans software Metal shadow stripe"];
                    id<MTLComputeCommandEncoder> shadowEncoder =
                        [commandBuffer computeCommandEncoder];
                    if (shadowEncoder == nil)
                    {
                        throw std::runtime_error(
                            "Metal failed to create the shadow compute encoder");
                    }
                    [shadowEncoder
                        setLabel:@"Evans Metal shadow stripe"];
                    [shadowEncoder
                        setComputePipelineState:ShadowPipeline];
                    [shadowEncoder setBuffer:buffers.ShadowTriangles
                                      offset:0
                                     atIndex:0];
                    [shadowEncoder setBuffer:buffers.ShadowOffsets
                                      offset:0
                                     atIndex:1];
                    [shadowEncoder
                        setBuffer:buffers.ShadowTriangleIndices
                           offset:0
                          atIndex:2];
                    [shadowEncoder setBytes:&constants
                                     length:sizeof(constants)
                                    atIndex:3];
                    [shadowEncoder setBuffer:buffers.ShadowDepths
                                      offset:0
                                     atIndex:4];
                    [shadowEncoder setBytes:&row
                                      length:sizeof(row)
                                     atIndex:5];
                    [shadowEncoder setBuffer:buffers.TextureDescriptors
                                      offset:0
                                     atIndex:6];
                    [shadowEncoder setBuffer:buffers.TextureTexels
                                      offset:0
                                     atIndex:7];
                    const std::uint32_t rowCount = std::min(
                        RowsPerSubmission,
                        shadowHeight - row);
                    const MTLSize threadgroups = MTLSizeMake(
                        (shadowWidth + 7U) / 8U,
                        (rowCount + 7U) / 8U,
                        1);
                    [shadowEncoder
                        dispatchThreadgroups:threadgroups
                        threadsPerThreadgroup:threadsPerGroup];
                    [shadowEncoder endEncoding];

                    std::string commandError;
                    if (!CommitAndWait(commandBuffer, commandError))
                    {
                        throw std::runtime_error(commandError);
                    }
                }
            }
        }

        if (Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }
        for (std::uint32_t row = 0;
             row < target.Height();
             row += RowsPerSubmission)
        {
            if (Detail::IsGpuRasterCancelled(
                    externalCancellation,
                    backendCancellation))
            {
                return RasterStatus::Cancelled;
            }
            @autoreleasepool
            {
                id<MTLCommandBuffer> commandBuffer =
                    [Queue commandBuffer];
                if (commandBuffer == nil)
                {
                    throw std::runtime_error(
                        "Metal failed to create a raster command buffer");
                }
                [commandBuffer
                    setLabel:@"Evans software Metal raster stripe"];
                id<MTLComputeCommandEncoder> rasterEncoder =
                    [commandBuffer computeCommandEncoder];
                if (rasterEncoder == nil)
                {
                    throw std::runtime_error(
                        "Metal failed to create the raster compute encoder");
                }
                [rasterEncoder
                    setLabel:@"Evans Metal raster stripe"];
                [rasterEncoder
                    setComputePipelineState:RasterPipeline];
                [rasterEncoder setBuffer:buffers.MainTriangles
                                  offset:0
                                 atIndex:0];
                [rasterEncoder setBuffer:buffers.MainOffsets
                                  offset:0
                                 atIndex:1];
                [rasterEncoder
                    setBuffer:buffers.MainTriangleIndices
                       offset:0
                      atIndex:2];
                [rasterEncoder setBuffer:buffers.ShadowDepths
                                  offset:0
                                 atIndex:3];
                [rasterEncoder setBytes:&constants
                                 length:sizeof(constants)
                                atIndex:4];
                [rasterEncoder setBuffer:buffers.OutputColors
                                  offset:0
                                 atIndex:5];
                [rasterEncoder setBuffer:buffers.OutputDepths
                                  offset:0
                                 atIndex:6];
                [rasterEncoder setBytes:&row
                                  length:sizeof(row)
                                 atIndex:7];
                [rasterEncoder setBuffer:buffers.TextureDescriptors
                                  offset:0
                                 atIndex:8];
                [rasterEncoder setBuffer:buffers.TextureTexels
                                  offset:0
                                 atIndex:9];
                const std::uint32_t rowCount = std::min(
                    RowsPerSubmission,
                    target.Height() - row);
                const MTLSize threadgroups = MTLSizeMake(
                    (target.Width() + 7U) / 8U,
                    (rowCount + 7U) / 8U,
                    1);
                [rasterEncoder
                    dispatchThreadgroups:threadgroups
                    threadsPerThreadgroup:threadsPerGroup];
                [rasterEncoder endEncoding];

                std::string commandError;
                if (!CommitAndWait(commandBuffer, commandError))
                {
                    throw std::runtime_error(commandError);
                }
            }
        }
        if (Detail::IsGpuRasterCancelled(
                externalCancellation,
                backendCancellation))
        {
            return RasterStatus::Cancelled;
        }

        const float* colors =
            static_cast<const float*>([buffers.OutputColors contents]);
        const float* depths =
            static_cast<const float*>([buffers.OutputDepths contents]);
        if (colors == nullptr || depths == nullptr)
        {
            throw std::runtime_error(
                "Metal raster output buffers are not CPU-visible");
        }

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

        Detail::ApplyGpuSelectionOutline(mainScene, gpuSettings, target,
            externalCancellation, backendCancellation);
        if (Detail::IsGpuRasterCancelled(externalCancellation, backendCancellation))
            return RasterStatus::Cancelled;

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
            "Native Metal compute, 16x16 bins=" +
            std::to_string(mainBins.TilesX * mainBins.TilesY) +
            ", refs=" +
            std::to_string(mainBins.TriangleIndices.size()) +
            ", max-list=" +
            std::to_string(mainBins.MaximumListLength) +
            (generateShadowMap
                 ? ", GPU shadow-depth pass"
                 : hasExternalShadowMap
                       ? ", uploaded shadow map"
                       : "") +
            ", device=" +
            DeviceName;
        statistics.RasterMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - rasterBegin)
                .count();
        return RasterStatus::Completed;
    }

    Detail::ExecutionGate Execution;
    Detail::StatisticsStore Statistics;
    mutable std::mutex StateMutex;
    std::string Error;
    std::string DeviceName;
    bool NativeAvailable = false;
    id<MTLDevice> Device = nil;
    id<MTLCommandQueue> Queue = nil;
    id<MTLComputePipelineState> ShadowPipeline = nil;
    id<MTLComputePipelineState> RasterPipeline = nil;
};

MetalRasterRHI::MetalRasterRHI()
    : Context_(std::make_unique<Context>())
{
    (void)Initialize({});
}

MetalRasterRHI::~MetalRasterRHI()
{
    Shutdown();
}

bool MetalRasterRHI::Initialize(const RasterRHICreateInfo& createInfo)
{
    Shutdown();
    std::string error;
    bool available = false;
    @autoreleasepool
    {
        available = Context_->InitializeNative(error);
    }
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        Context_->NativeAvailable = available;
        Context_->Error = std::move(error);
    }
    if (!available)
    {
        return false;
    }
    return Context_->Execution.Initialize(createInfo.WorkerThreads);
}

void MetalRasterRHI::Shutdown() noexcept
{
    Context_->Execution.Shutdown();
    {
        std::lock_guard<std::mutex> lock(Context_->StateMutex);
        Context_->NativeAvailable = false;
        @autoreleasepool
        {
            Context_->ShutdownNative();
        }
    }
}

bool MetalRasterRHI::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return Context_->NativeAvailable &&
           Context_->Execution.IsInitialized();
}

RasterBackend MetalRasterRHI::Backend() const noexcept
{
    return RasterBackend::Metal;
}

std::string_view MetalRasterRHI::Name() const noexcept
{
    return "Metal Compute Raster";
}

RasterBackendCapabilities MetalRasterRHI::Capabilities() const
{
    std::lock_guard<std::mutex> lock(Context_->StateMutex);
    return {
        RasterBackend::Metal,
        Context_->NativeAvailable,
        Context_->NativeAvailable,
        8,
        8,
        Context_->NativeAvailable
            ? "Native Metal tiled compute raster and shadow kernels on " +
                  Context_->DeviceName
            : "Unsupported: " + Context_->Error};
}

RasterStatus MetalRasterRHI::Render(
    const RasterScene& scene,
    const RasterSettings& settings,
    RenderTarget& target,
    const RasterCancellationToken& cancellation)
{
    Detail::ValidateRenderTarget(target);
    Context_->ThrowIfNativeUnavailable();

    RasterStatistics statistics;
    const RasterStatus status = Context_->Execution.Execute(
        [&](const std::atomic<bool>& backendCancellation)
        {
            Context_->ThrowIfNativeUnavailable();
            @autoreleasepool
            {
                return Context_->RenderNative(
                    scene,
                    settings,
                    target,
                    cancellation,
                    backendCancellation,
                    statistics);
            }
        });
    Context_->Statistics.Set(std::move(statistics));
    return status;
}

void MetalRasterRHI::Cancel() noexcept
{
    Context_->Execution.Cancel();
}

void MetalRasterRHI::WaitIdle() noexcept
{
    Context_->Execution.WaitIdle();
}

RasterStatistics MetalRasterRHI::LastStatistics() const
{
    return Context_->Statistics.Get();
}

} // namespace Evans::Rasterizer
