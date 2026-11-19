// Thin-lens circles of confusion are affine in hardware depth. Their signed
// radii distinguish foreground coverage from background blur. This deterministic
// integer-disk gather is a readable reference, not a layered visibility solution.
// Demers, GPU Gems 23, "Depth of Field: A Survey of Techniques" (2004):
// https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-23-depth-field-survey-techniques
// Hammon, GPU Gems 3, "Practical Post-Process Depth of Field" (2007), discusses
// foreground spreading and the missing background behind a single depth layer:
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
cbuffer DepthOfFieldConstants : register(b0)
{
    float4 ViewportRadius;
    float4 CoCScaleBias;
    uint4 Options;
};
Texture2D<float4> SourceColor : register(t0);
Texture2D<float> OpaqueDepth : register(t1);

static const uint CompositePass = 1u;
static const float ClearDepth = 1.0f;
static const float Pi = 3.141592653589793f;
static const float PixelFootprintHalfWidth = 0.5f;
static const float SharpRadius = 0.5f;
static const float FullBlurRadius = 1.5f;
static const float MinimumDiskArea = 1.0f;
static const float MinimumAccumulatedWeight = 1.0e-7f;
static const float MaximumHalfFloatRadiance = 65504.0f;

struct ScreenVertex { float4 Position : SV_POSITION; };
ScreenVertex VSMain(uint vertexId : SV_VertexID)
{
    float2 corner = float2((vertexId << 1u) & 2u, vertexId & 2u);
    ScreenVertex output;
    output.Position = float4(corner * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}

float CircleOfConfusion(float depth)
{
    float radius = depth >= ClearDepth ? CoCScaleBias.z : depth * CoCScaleBias.x + CoCScaleBias.y;
    return clamp(radius, -ViewportRadius.z, ViewportRadius.z);
}

float BlurAmount(float radius)
{
    return smoothstep(SharpRadius, FullBlurRadius, radius);
}

float DiskWeight(float radius, float distance)
{
    float coverage = saturate(radius + PixelFootprintHalfWidth - distance);
    return coverage / max(Pi * radius * radius, MinimumDiskArea);
}

float4 PSMain(ScreenVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    float4 center = SourceColor.Load(int3(pixel, 0));
    if (Options.x == CompositePass || ViewportRadius.z <= 0.0f || CoCScaleBias.w <= 0.0f ||
        (CoCScaleBias.x == 0.0f && CoCScaleBias.y == 0.0f)) return center;

    uint width, height;
    SourceColor.GetDimensions(width, height);
    int2 extent = int2(width, height);
    float centerRadius = CircleOfConfusion(OpaqueDepth.Load(int3(pixel, 0)));
    float searchRadius = ViewportRadius.z + PixelFootprintHalfWidth;
    int searchExtent = int(ceil(searchRadius));
    float3 nearColor = 0.0f, farColor = 0.0f;
    float nearWeight = 0.0f, farWeight = 0.0f;
    [loop] for (int y = -searchExtent; y <= searchExtent; ++y)
    {
        [loop] for (int x = -searchExtent; x <= searchExtent; ++x)
        {
            float distance = sqrt(float(x * x + y * y));
            if (distance > searchRadius) continue;
            // Edge extension retains the kernel's original offset and coverage.
            int2 neighbor = clamp(pixel + int2(x, y), int2(0, 0), extent - 1);
            float radius = CircleOfConfusion(OpaqueDepth.Load(int3(neighbor, 0)));
            float3 radiance = SourceColor.Load(int3(neighbor, 0)).rgb;
            if (radius < 0.0f)
            {
                float weight = DiskWeight(-radius, distance) * BlurAmount(-radius);
                nearColor += radiance * weight;
                nearWeight += weight;
            }
            else
            {
                // Only available background neighbors can fill behind a blurred
                // foreground silhouette; hidden surfaces are absent from this image.
                float gatherRadius = centerRadius < 0.0f ? max(radius, -centerRadius) : radius;
                float weight = DiskWeight(gatherRadius, distance);
                farColor += radiance * weight;
                farWeight += weight;
            }
        }
    }
    float3 result = center.rgb;
    if (farWeight > MinimumAccumulatedWeight)
        result = lerp(result, farColor / farWeight, BlurAmount(abs(centerRadius)));
    if (nearWeight > MinimumAccumulatedWeight)
        result = lerp(result, nearColor / nearWeight, saturate(nearWeight));
    // The composite pass writes RGB only, retaining the scene's original alpha.
    return float4(min(result, MaximumHalfFloatRadiance), center.a);
}
