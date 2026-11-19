// DX12 contact shadows, authored independently for this native backend.
// McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing" (2014):
// https://jcgt.org/published/0003/04/04/paper.pdf
// Traverse exact screen-cell intervals and intersect finite axial-depth slabs.
// Interpolating depth/clip-W and reciprocal clip-W preserves perspective.
// This pass produces raw selected-light visibility, never shaded scene color.
#if defined(EVANS_CONTACT_DEPTH)
// Depth-only draw uses the DX12 surface ABI and the same alpha/UV convention.
// Camera coverage is captured before tracing, independent of HDR lighting.
cbuffer CameraDepthDraw : register(b0)
{
    row_major float4x4 ObjectToWorld, ObjectNormal, Camera, LightCamera;
    float4 DrawColor, DiffuseColor, AmbientColor, SpecularColor, EmissionColor;
    float4 Material, AmbientLight, CameraLights, DrawSettings;
    uint4 MaterialImages;
};
Texture2D<float4> DiffuseTexture : register(t1);
Texture2D<float4> OpacityTexture : register(t3);
SamplerState MaterialSampler : register(s0);
struct DepthInput { float3 Position : POSITION; float3 Normal : NORMAL; float4 Color : COLOR0; float2 UV : TEXCOORD0; };
// DX12's instanced forward vertex entry shares this full varying signature.
// Keeping the unused fields preserves COLOR/UV register locations for the PSO.
struct DepthFragment
{
    float4 Position : SV_Position;
    float3 World : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 ShadowPosition : TEXCOORD2;
    float2 UV : TEXCOORD3;
};
DepthFragment VSDepth(DepthInput vertex)
{
    DepthFragment result = (DepthFragment)0;
    result.Position = mul(Camera, mul(ObjectToWorld, float4(vertex.Position, 1.0f)));
    result.Color = vertex.Color; result.UV = float2(vertex.UV.x, 1.0f - vertex.UV.y);
    return result;
}
void PSDepth(DepthFragment fragment)
{
    float diffuseAlpha = MaterialImages.x != 0 ? DiffuseTexture.Sample(MaterialSampler, fragment.UV).a : 1.0f;
    float coverage = MaterialImages.z != 0 ? OpacityTexture.Sample(MaterialSampler, fragment.UV).r : 1.0f;
    float opacity = saturate(fragment.Color.a * Material.y * diffuseAlpha * coverage);
    if (Material.z > 0.0f) clip(opacity - Material.z);
}
#else
#ifndef EVANS_CONTACT_MULTISAMPLE
#define EVANS_CONTACT_MULTISAMPLE 0
#endif

cbuffer TraceDescription : register(b0)
{
    row_major float4x4 CameraProjection;
    row_major float4x4 WorldFromClip;
    float4 CameraDepth; // near, far, perspective flag, minimum positive W
    float4 EmitterPositionKind; // zero kind denotes a directional emitter
    float4 TowardEmitter;
    float4 RaySettings; // distance limit, start bias, slab thickness, reserved
    uint4 RasterExtentBudget; // width, height, visited-cell budget, sample count
};
#if EVANS_CONTACT_MULTISAMPLE
Texture2DMS<float> OpaqueDepth : register(t0);
#else
Texture2D<float> OpaqueDepth : register(t0);
#endif
RWTexture2DArray<float> ContactVisibility : register(u0);

static const uint GroupWidth = 8u;
static const uint MaximumCells = 256u;
static const float HalfPixel = 0.5f;
static const float BackgroundDepth = 1.0f;
static const float SmallPixelDirection = 1.0e-6f;
static const float LargestFloat = 3.402823466e+38f;
static const float SinglePrecisionEpsilon = 1.1920928955078125e-7f;
static const float CornerTolerance = 4.0f * SinglePrecisionEpsilon;
static const float AxialRoundoff = 4.0f * SinglePrecisionEpsilon;

float FetchDepth(int2 cell, uint sampleIndex)
{
#if EVANS_CONTACT_MULTISAMPLE
    return OpaqueDepth.Load(cell, sampleIndex);
#else
    return OpaqueDepth.Load(int3(cell, 0));
#endif
}
float AxialDistance(float depth)
{
    return CameraDepth.z > HalfPixel
        ? CameraDepth.x * CameraDepth.y / (CameraDepth.y - depth * (CameraDepth.y - CameraDepth.x))
        : lerp(CameraDepth.x, CameraDepth.y, depth);
}
bool LimitSegment(float startDistance, float endDistance, inout float2 interval)
{
    if (startDistance < 0.0f && endDistance < 0.0f) return false;
    if ((startDistance < 0.0f) != (endDistance < 0.0f))
    {
        float crossing = startDistance / (startDistance - endDistance);
        if (startDistance < 0.0f) interval.x = max(interval.x, crossing);
        else interval.y = min(interval.y, crossing);
    }
    return interval.x < interval.y;
}
bool ClipCameraSegment(float4 startClip, float4 endClip, out float2 interval)
{
    interval = float2(0.0f, 1.0f);
    return LimitSegment(startClip.w - CameraDepth.w, endClip.w - CameraDepth.w, interval)
        && LimitSegment(startClip.w + startClip.x, endClip.w + endClip.x, interval)
        && LimitSegment(startClip.w - startClip.x, endClip.w - endClip.x, interval)
        && LimitSegment(startClip.w + startClip.y, endClip.w + endClip.y, interval)
        && LimitSegment(startClip.w - startClip.y, endClip.w - endClip.y, interval)
        && LimitSegment(startClip.z, endClip.z, interval)
        && LimitSegment(startClip.w - startClip.z, endClip.w - endClip.z, interval);
}
float2 ToPixel(float4 clip)
{
    return (float2(clip.x, -clip.y) / clip.w * HalfPixel + HalfPixel) * RasterExtentBudget.xy;
}
float DepthOnSegment(float progress, float2 weightedDepth, float2 reciprocalW)
{
    return lerp(weightedDepth.x, weightedDepth.y, progress) / lerp(reciprocalW.x, reciprocalW.y, progress);
}

float TraceVisibility(float3 receiver, int2 receiverCell, uint sampleIndex)
{
    bool directional = EmitterPositionKind.w == 0.0f;
    float3 displacement = directional ? TowardEmitter.xyz : EmitterPositionKind.xyz - receiver;
    float distanceToLight = length(displacement);
    if (distanceToLight <= 0.0f) return 1.0f;
    float rayLength = directional ? RaySettings.x : min(RaySettings.x, distanceToLight);
    if (rayLength <= RaySettings.y) return 1.0f;
    float3 direction = displacement / distanceToLight;
    float3 first = receiver + direction * RaySettings.y;
    float3 last = receiver + direction * rayLength;
    float4 firstClip = mul(CameraProjection, float4(first, 1.0f));
    float4 lastClip = mul(CameraProjection, float4(last, 1.0f));
    float2 cameraInterval;
    if (!ClipCameraSegment(firstClip, lastClip, cameraInterval)) return 1.0f;
    // Project the clipped world endpoints before dividing by homogeneous W.
    float3 clippedFirst = lerp(first, last, cameraInterval.x);
    float3 clippedLast = lerp(first, last, cameraInterval.y);
    firstClip = mul(CameraProjection, float4(clippedFirst, 1.0f));
    lastClip = mul(CameraProjection, float4(clippedLast, 1.0f));
    float2 origin = ToPixel(firstClip);
    float2 delta = ToPixel(lastClip) - origin;
    if (max(abs(delta.x), abs(delta.y)) <= SmallPixelDirection) return 1.0f;
    // Convert the viewport's float roundoff to this clipped segment's scale.
    float intervalUncertainty = CornerTolerance * float(max(RasterExtentBudget.x, RasterExtentBudget.y))
        / max(abs(delta.x), abs(delta.y));
    // Compare crossing times with the sum of their per-axis error bounds.
    float minorDisplacement = min(abs(delta.x), abs(delta.y));
    float crossingUncertainty = intervalUncertainty + (minorDisplacement > SmallPixelDirection ?
        CornerTolerance * float(max(RasterExtentBudget.x, RasterExtentBudget.y)) / minorDisplacement : 0.0f);

    int2 cell = int2(floor(origin));
    int2 advance = int2(delta.x < 0.0f ? -1 : 1, delta.y < 0.0f ? -1 : 1);
    float2 nextCrossing, crossingStride;
    [unroll] for (uint axis = 0; axis < 2u; ++axis)
    {
        if (delta[axis] < 0.0f && origin[axis] == floor(origin[axis])) --cell[axis];
        cell[axis] = clamp(cell[axis], 0, int(RasterExtentBudget[axis]) - 1);
        if (abs(delta[axis]) <= SmallPixelDirection)
            nextCrossing[axis] = crossingStride[axis] = LargestFloat;
        else
        {
            float edge = float(cell[axis] + (advance[axis] > 0 ? 1 : 0));
            nextCrossing[axis] = max(0.0f, (edge - origin[axis]) / delta[axis]);
            crossingStride[axis] = 1.0f / abs(delta[axis]);
        }
    }
    float2 inverseW = 1.0f / float2(firstClip.w, lastClip.w);
    float2 depthOverW = float2(AxialDistance(firstClip.z * inverseW.x), AxialDistance(lastClip.z * inverseW.y)) * inverseW;
    float intervalStart = 0.0f;
    [loop] for (uint visited = 0; visited < min(RasterExtentBudget.z, MaximumCells) && intervalStart < 1.0f; ++visited)
    {
        if (any(cell < 0) || any(cell >= int2(RasterExtentBudget.xy))) break;
        float intervalEnd = min(1.0f, min(nextCrossing.x, nextCrossing.y));
        float storedDepth = FetchDepth(cell, sampleIndex);
        if (intervalEnd > intervalStart + intervalUncertainty && any(cell != receiverCell) && storedDepth < BackgroundDepth)
        {
            float front = AxialDistance(storedDepth);
            float back = front + RaySettings.z;
            float startDepth = DepthOnSegment(intervalStart, depthOverW, inverseW);
            float endDepth = DepthOnSegment(intervalEnd, depthOverW, inverseW);
            float depthError = AxialRoundoff * max(1.0f,
                max(max(abs(front), abs(back)), max(abs(startDepth), abs(endDepth))));
            // Positive clip W makes axial depth monotone on a cell interval.
            // Require interval overlap beyond native-float uncertainty.
            if (max(startDepth, endDepth) >= front + depthError && min(startDepth, endDepth) <= back - depthError) return 0.0f;
        }
        if (intervalEnd >= 1.0f) break;
        intervalStart = max(intervalStart, intervalEnd);
        [unroll] for (uint axis = 0; axis < 2u; ++axis)
            if (nextCrossing[axis] <= intervalEnd + crossingUncertainty)
            {
                intervalStart = max(intervalStart, nextCrossing[axis]);
                cell[axis] += advance[axis];
                nextCrossing[axis] += crossingStride[axis];
            }
    }
    // Screen exits and budget exhaustion are unknown, not occluded.
    return 1.0f;
}

[numthreads(GroupWidth, GroupWidth, 1)]
void CSMain(uint3 invocation : SV_DispatchThreadID)
{
    if (any(invocation.xy >= RasterExtentBudget.xy) || invocation.z >= RasterExtentBudget.w) return;
    float depth = FetchDepth(int2(invocation.xy), invocation.z);
    float visibility = 1.0f;
    if (depth < BackgroundDepth)
    {
        float2 sampleOffset = 0.0f;
#if EVANS_CONTACT_MULTISAMPLE
        sampleOffset = OpaqueDepth.GetSamplePosition(invocation.z);
#endif
        float2 ndc = (float2(invocation.xy) + HalfPixel + sampleOffset) / RasterExtentBudget.xy * 2.0f - 1.0f;
        float4 world = mul(WorldFromClip, float4(ndc.x, -ndc.y, depth, 1.0f));
        visibility = TraceVisibility(world.xyz / world.w, int2(invocation.xy), invocation.z);
    }
    ContactVisibility[invocation] = visibility;
}
#endif
