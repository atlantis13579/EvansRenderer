// Weighted Blended Order-Independent Transparency, McGuire and Bavoil (2013).
// https://jcgt.org/published/0002/02/09/
// DX12's two-MRT surface pass stores a weighted radiance sum and multiplicative
// background visibility. This standalone resolve emits straight radiance and
// coverage for fixed-function source-over blending into scene-linear HDR.
Texture2D<float4> RadianceSum : register(t0);
Texture2D<float> BackgroundVisibility : register(t1);
Texture2DMS<float4> SampleRadianceSum : register(t2);
Texture2DMS<float> SampleBackgroundVisibility : register(t3);

struct ResolveVertex { float4 Position : SV_Position; };
ResolveVertex VSMain(uint vertexId : SV_VertexID)
{
    const float OverscanCorner = 3.0;
    ResolveVertex vertex;
    vertex.Position = float4(vertexId == 1 ? OverscanCorner : -1.0,
        vertexId == 2 ? OverscanCorner : -1.0, 0.0, 1.0);
    return vertex;
}

float4 CompositeLayer(float4 sum, float backgroundVisibility)
{
    if (sum.w <= 0.0) return 0.0; // Empty pixels leave the scene unchanged.
    const float SceneRadianceLimit = 65504.0; // Largest finite RGBA16F channel.
    float3 averageRadiance = clamp(sum.xyz / sum.w, 0.0, SceneRadianceLimit);
    return float4(averageRadiance, 1.0 - saturate(backgroundVisibility));
}

float4 PSMain(ResolveVertex vertex) : SV_Target0
{
    int3 location = int3(vertex.Position.xy, 0);
    return CompositeLayer(RadianceSum.Load(location), BackgroundVisibility.Load(location));
}

float4 PSResolveSamples(ResolveVertex vertex, uint sampleId : SV_SampleIndex) : SV_Target0
{
    // Run at sample frequency: averaging statistics before this nonlinear
    // normalization would destroy correlated silhouette coverage.
    int2 location = int2(vertex.Position.xy);
    return CompositeLayer(SampleRadianceSum.Load(location, sampleId),
        SampleBackgroundVisibility.Load(location, sampleId));
}
