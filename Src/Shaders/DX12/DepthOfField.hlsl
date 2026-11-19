// Independently authored DX12 thin-lens depth of field. CPU camera constants
// convert hardware depth to signed pixel radii; foreground disks cover neighbors
// while the background layer is gathered separately. The integer disk is a
// deterministic spatial reference without history or additional scene assets.
// Demers, "Depth of Field: A Survey of Techniques", GPU Gems 23 (2004):
// https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-23-depth-field-survey-techniques
// Foreground visibility limitations: Hammon, GPU Gems 3, chapter 28 (2007):
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
cbuffer LensConstants : register(b0)
{
    float4 InverseExtentMaximumRadius;
    float4 DepthCircleCoefficients;
    uint4 StageOptions;
};
Texture2D<float4> InputRadiance : register(t0);
Texture2D<float> SceneDepth : register(t1);

static const uint CompositeStage = 1u;
static const float BackgroundDepth = 1.0f;
static const float Pi = 3.141592653589793f;
static const float PixelHalfExtent = 0.5f;
static const float SharpCircleRadius = 0.5f;
static const float BlurredCircleRadius = 1.5f;
static const float UnitPixelArea = 1.0f;
static const float MinimumLayerWeight = 1.0e-7f;
static const float MaximumHalfFloatRadiance = 65504.0f;

struct FullscreenVertex { float4 Position : SV_POSITION; };
FullscreenVertex VSMain(uint id : SV_VertexID)
{
    float2 coordinate = float2((id << 1u) & 2u, id & 2u);
    FullscreenVertex vertex;
    vertex.Position = float4(coordinate.x * 2.0f - 1.0f, 1.0f - coordinate.y * 2.0f, 0.0f, 1.0f);
    return vertex;
}

float SignedRadius(int2 texel)
{
    float depth = SceneDepth.Load(int3(texel, 0));
    float radius = depth >= BackgroundDepth ? DepthCircleCoefficients.z :
        DepthCircleCoefficients.x * depth + DepthCircleCoefficients.y;
    return clamp(radius, -InverseExtentMaximumRadius.z, InverseExtentMaximumRadius.z);
}

float RadiusBlend(float radius)
{
    return smoothstep(SharpCircleRadius, BlurredCircleRadius, radius);
}

float DiskCoverage(float radius, float pixelDistance)
{
    float boundaryCoverage = saturate(radius + PixelHalfExtent - pixelDistance);
    float diskArea = max(Pi * radius * radius, UnitPixelArea);
    return boundaryCoverage / diskArea;
}

float3 GatherLayers(int2 centerPixel, int2 dimensions, float3 centerColor)
{
    float centerCircle = SignedRadius(centerPixel);
    float supportRadius = InverseExtentMaximumRadius.z + PixelHalfExtent;
    int supportExtent = int(ceil(supportRadius));
    float3 foreground = 0.0f;
    float3 background = 0.0f;
    float foregroundWeight = 0.0f;
    float backgroundWeight = 0.0f;
    [loop] for (int offsetY = -supportExtent; offsetY <= supportExtent; ++offsetY)
    {
        [loop] for (int offsetX = -supportExtent; offsetX <= supportExtent; ++offsetX)
        {
            int2 offset = int2(offsetX, offsetY);
            float offsetLength = sqrt(float(offsetX * offsetX + offsetY * offsetY));
            if (offsetLength > supportRadius) continue;
            // Clamped fetches extend border colors, not the disk's sample distance.
            int2 samplePixel = clamp(centerPixel + offset, int2(0, 0), dimensions - 1);
            float circle = SignedRadius(samplePixel);
            float3 color = InputRadiance.Load(int3(samplePixel, 0)).rgb;
            if (circle < 0.0f)
            {
                float nearRadius = -circle;
                float contribution = DiskCoverage(nearRadius, offsetLength) * RadiusBlend(nearRadius);
                foreground += color * contribution;
                foregroundWeight += contribution;
            }
            else
            {
                // A single depth layer cannot reveal hidden background. Visible
                // background samples provide the local fill under a near-field blur.
                float radius = centerCircle < 0.0f ? max(circle, -centerCircle) : circle;
                float contribution = DiskCoverage(radius, offsetLength);
                background += color * contribution;
                backgroundWeight += contribution;
            }
        }
    }
    float3 color = centerColor;
    if (backgroundWeight > MinimumLayerWeight)
        color = lerp(color, background / backgroundWeight, RadiusBlend(abs(centerCircle)));
    if (foregroundWeight > MinimumLayerWeight)
        color = lerp(color, foreground / foregroundWeight, saturate(foregroundWeight));
    return min(color, MaximumHalfFloatRadiance);
}

float4 PSMain(FullscreenVertex fragment) : SV_TARGET
{
    int2 texel = int2(fragment.Position.xy);
    float4 center = InputRadiance.Load(int3(texel, 0));
    if (StageOptions.x == CompositeStage || InverseExtentMaximumRadius.z <= 0.0f ||
        DepthCircleCoefficients.w <= 0.0f ||
        (DepthCircleCoefficients.x == 0.0f && DepthCircleCoefficients.y == 0.0f)) return center;
    uint width, height;
    InputRadiance.GetDimensions(width, height);
    // Composite uses an RGB-only target write mask; alpha remains geometrical coverage.
    return float4(GatherLayers(texel, int2(width, height), center.rgb), center.a);
}
