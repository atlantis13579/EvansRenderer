// DX11 contact visibility: an independent native HLSL implementation.
// McGuire and Mara, "Efficient GPU Screen-Space Ray Tracing" (2014):
// https://jcgt.org/published/0003/04/04/paper.pdf
// Perspective-correct axial depth is interpolated over exact pixel-cell
// intervals. Each visible sample represents a finite slab, not an infinite
// extrusion. Unknown/off-screen layers and exhausted budgets stay unoccluded.
// The selected light's receiver multiplies this raw mask by raw primary-shadow
// visibility before applying artistic strength. This is not a color post-effect.
#if defined(EVANS_CONTACT_DEPTH)
// The prepass reproduces the main surface's alpha-tested coverage. It uses the
// existing draw ABI, with no lighting, HDR targets or light-space depth bias.
cbuffer DepthDrawConstants : register(b0)
{
    row_major float4x4 World, NormalWorld, ViewProjection, LightViewProjection;
    float4 PipelineColor, MaterialDiffuseColor, MaterialAmbientColor, MaterialSpecularColor, MaterialEmissiveColor;
    float4 MaterialParameters, AmbientColorStrength, CameraLightCount, RenderParameters;
    uint4 TextureFlags;
};
Texture2D<float4> DiffuseMap : register(t1);
Texture2D<float4> OpacityMap : register(t3);
SamplerState SurfaceSampler : register(s0);
struct DepthVertexInput { float3 Position : POSITION; float3 Normal : NORMAL; float4 Color : COLOR0; float2 UV : TEXCOORD0; };
// Preserve the complete forward varying layout: the shared instanced vertex
// program emits COLOR and UV at these physical signature registers. Unused
// interpolants cannot be removed from this declaration without breaking linkage.
struct DepthVertex
{
    float4 Position : SV_Position;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 LightPosition : TEXCOORD2;
    float2 UV : TEXCOORD3;
};
DepthVertex VSDepth(DepthVertexInput input)
{
    DepthVertex output = (DepthVertex)0;
    output.Position = mul(ViewProjection, mul(World, float4(input.Position, 1.0f)));
    output.Color = input.Color; output.UV = float2(input.UV.x, 1.0f - input.UV.y);
    return output;
}
void PSDepth(DepthVertex input)
{
    float diffuseAlpha = TextureFlags.x != 0 ? DiffuseMap.Sample(SurfaceSampler, input.UV).a : 1.0f;
    float opacity = TextureFlags.z != 0 ? OpacityMap.Sample(SurfaceSampler, input.UV).r : 1.0f;
    float alpha = saturate(input.Color.a * MaterialParameters.y * diffuseAlpha * opacity);
    if (MaterialParameters.z > 0.0f) clip(alpha - MaterialParameters.z);
}
#else
#ifndef EVANS_CONTACT_MULTISAMPLE
#define EVANS_CONTACT_MULTISAMPLE 0
#endif
static const uint ThreadsPerAxis = 8u;
static const uint MaximumTraceSteps = 256u;
static const float PixelCenter = 0.5f;
static const float PixelDirectionEpsilon = 1.0e-6f;
static const float FloatPrecision = 1.1920928955078125e-7f;
static const float CrossingToleranceUlps = 4.0f;
static const float CrossingTolerance = CrossingToleranceUlps * FloatPrecision;
static const float DepthUncertaintyUlps = 4.0f;
static const float NoCrossing = 3.402823466e+38f;
static const float ClearDepth = 1.0f;

cbuffer ContactConstants : register(b0)
{
    row_major float4x4 ViewProjection;
    row_major float4x4 InverseViewProjection;
    float4 DepthMetric;
    float4 LightPositionType;
    float4 LightDirection;
    float4 TraceParameters;
    uint4 ImageAndBudget;
};
#if EVANS_CONTACT_MULTISAMPLE
Texture2DMS<float> SceneDepth : register(t0);
#else
Texture2D<float> SceneDepth : register(t0);
#endif
RWTexture2DArray<float> ContactVisibility : register(u0);

float ReadDepth(int2 pixel, uint sampleIndex)
{
#if EVANS_CONTACT_MULTISAMPLE
    return SceneDepth.Load(pixel, sampleIndex);
#else
    return SceneDepth.Load(int3(pixel, 0));
#endif
}
float LinearDepth(float stored)
{
    return DepthMetric.z != 0.0f ? rcp((1.0f - stored) / DepthMetric.x + stored / DepthMetric.y) : stored * DepthMetric.y;
}
bool ClipPlane(float first, float last, inout float enter, inout float leave)
{
    if (first < 0.0f && last < 0.0f) return false;
    if ((first < 0.0f) != (last < 0.0f))
    {
        float crossing = first / (first - last);
        if (first < 0.0f) enter = max(enter, crossing);
        else leave = min(leave, crossing);
    }
    return enter < leave;
}
float TraceVisibility(float3 receiver, float3 endpoint, int2 receiverPixel, uint sampleIndex)
{
    float rayLength = length(endpoint - receiver);
    if (rayLength <= TraceParameters.y) return 1.0f;
    float3 first = lerp(receiver, endpoint, TraceParameters.y / rayLength);
    float4 h0 = mul(ViewProjection, float4(first, 1.0f));
    float4 h1 = mul(ViewProjection, float4(endpoint, 1.0f));
    float enter = 0.0f, leave = 1.0f;
    if (!ClipPlane(h0.w - DepthMetric.w, h1.w - DepthMetric.w, enter, leave) ||
        !ClipPlane(h0.w + h0.x, h1.w + h1.x, enter, leave) ||
        !ClipPlane(h0.w - h0.x, h1.w - h1.x, enter, leave) ||
        !ClipPlane(h0.w + h0.y, h1.w + h1.y, enter, leave) ||
        !ClipPlane(h0.w - h0.y, h1.w - h1.y, enter, leave) ||
        !ClipPlane(h0.z, h1.z, enter, leave) ||
        !ClipPlane(h0.w - h0.z, h1.w - h1.z, enter, leave)) return 1.0f;
    float4 lastClip = lerp(h0, h1, leave);
    h0 = lerp(h0, h1, enter); h1 = lastClip;
    float k0 = rcp(h0.w), k1 = rcp(h1.w);
    float2 extent = float2(ImageAndBudget.xy);
    float2 p0 = (h0.xy * k0 * float2(PixelCenter, -PixelCenter) + PixelCenter) * extent;
    float2 p1 = (h1.xy * k1 * float2(PixelCenter, -PixelCenter) + PixelCenter) * extent;
    float2 delta = p1 - p0;
    if (max(abs(delta.x), abs(delta.y)) <= PixelDirectionEpsilon) return 1.0f;
    // Keep uncertainty in pixel units even when clipping leaves a short ray.
    float progressTolerance = CrossingTolerance * max(extent.x, extent.y) / max(abs(delta.x), abs(delta.y));
    // Each axis has its own crossing-time uncertainty, especially on steep rays.
    float minorMotion = min(abs(delta.x), abs(delta.y));
    float cornerTolerance = progressTolerance + (minorMotion > PixelDirectionEpsilon ?
        CrossingTolerance * max(extent.x, extent.y) / minorMotion : 0.0f);
    int2 cell = int2(floor(p0));
    int2 step = int2(delta.x < 0.0f ? -1 : 1, delta.y < 0.0f ? -1 : 1);
    if (delta.x < 0.0f && p0.x == floor(p0.x)) --cell.x;
    if (delta.y < 0.0f && p0.y == floor(p0.y)) --cell.y;
    cell = clamp(cell, int2(0, 0), int2(ImageAndBudget.xy) - 1);
    float2 crossing = NoCrossing, increment = NoCrossing;
    if (abs(delta.x) > PixelDirectionEpsilon)
    {
        crossing.x = max(0.0f, (float(cell.x + (step.x > 0 ? 1 : 0)) - p0.x) / delta.x);
        increment.x = rcp(abs(delta.x));
    }
    if (abs(delta.y) > PixelDirectionEpsilon)
    {
        crossing.y = max(0.0f, (float(cell.y + (step.y > 0 ? 1 : 0)) - p0.y) / delta.y);
        increment.y = rcp(abs(delta.y));
    }
    float q0 = LinearDepth(h0.z * k0) * k0, q1 = LinearDepth(h1.z * k1) * k1;
    float begin = 0.0f;
    [loop] for (uint visited = 0u; visited < min(ImageAndBudget.z, MaximumTraceSteps); ++visited)
    {
        if (any(cell < 0) || any(cell >= int2(ImageAndBudget.xy))) break;
        float finish = min(min(crossing.x, crossing.y), 1.0f);
        float stored = ReadDepth(cell, sampleIndex);
        if (finish > begin + progressTolerance && any(cell != receiverPixel) && stored < ClearDepth)
        {
            float front = LinearDepth(stored), back = front + TraceParameters.z;
            float atBegin = lerp(q0, q1, begin) / lerp(k0, k1, begin);
            float atEnd = lerp(q0, q1, finish) / lerp(k0, k1, finish);
            float axialTolerance = DepthUncertaintyUlps * FloatPrecision *
                max(1.0f, max(max(abs(front), abs(back)), max(abs(atBegin), abs(atEnd))));
            // Axial depth is monotonic over a clipped straight segment: interval
            // overlap beyond roundoff gives visibility without solving a root.
            if (max(atBegin, atEnd) >= front + axialTolerance && min(atBegin, atEnd) <= back - axialTolerance) return 0.0f;
        }
        if (finish >= 1.0f) break;
        // Match the reference's half-open corner ownership at float precision.
        // A vanishing interval must not invent an extra blocker or consume budget.
        begin = max(begin, finish);
        if (crossing.x <= finish + cornerTolerance) { begin = max(begin, crossing.x); cell.x += step.x; crossing.x += increment.x; }
        if (crossing.y <= finish + cornerTolerance) { begin = max(begin, crossing.y); cell.y += step.y; crossing.y += increment.y; }
    }
    return 1.0f;
}

[numthreads(ThreadsPerAxis, ThreadsPerAxis, 1)]
void CSMain(uint3 dispatch : SV_DispatchThreadID)
{
    if (any(dispatch.xy >= ImageAndBudget.xy) || dispatch.z >= ImageAndBudget.w) return;
    int2 pixel = int2(dispatch.xy);
    float depth = ReadDepth(pixel, dispatch.z);
    float visibility = 1.0f;
    if (isfinite(depth) && depth >= 0.0f && depth < ClearDepth)
    {
        float2 sampleOffset = 0.0f;
#if EVANS_CONTACT_MULTISAMPLE
        // Native sample positions are relative to the pixel centre. This matches
        // DX11 Deferred/MSAA reconstruction and avoids bending sloped silhouettes.
        // https://learn.microsoft.com/windows/win32/direct3dhlsl/sm5-object-texture2dms-getsampleposition
        sampleOffset = SceneDepth.GetSamplePosition(dispatch.z);
#endif
        float2 uv = (float2(pixel) + PixelCenter + sampleOffset) / float2(ImageAndBudget.xy);
        float4 world = mul(InverseViewProjection, float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f));
        if (world.w != 0.0f)
        {
            float3 receiver = world.xyz / world.w;
            float3 towardLight = LightPositionType.w == 0.0f ? LightDirection.xyz : LightPositionType.xyz - receiver;
            float lightDistance = length(towardLight);
            if (lightDistance > 0.0f && all(isfinite(receiver)) && isfinite(lightDistance))
            {
                float distance = LightPositionType.w == 0.0f ? TraceParameters.x : min(TraceParameters.x, lightDistance);
                visibility = TraceVisibility(receiver, receiver + towardLight * (distance / lightDistance), pixel, dispatch.z);
            }
        }
    }
    ContactVisibility[dispatch] = visibility;
}
#endif
