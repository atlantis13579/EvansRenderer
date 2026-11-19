// DX11 multisample resolves. Color uses the API's linear-space hardware resolve.
// A single representative depth layer feeds screen-space effects: select the
// nearest covered sample, then fetch material and velocity from that same sample.
// Averaging normals or depths would invent surfaces at foreground silhouettes.
// https://learn.microsoft.com/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules
Texture2D<float4> ResolvedColor : register(t0);
Texture2DMS<float> SampleDepth : register(t1);
Texture2DMS<float4> DiffuseOpacity : register(t2);
Texture2DMS<float4> NormalRoughness : register(t3);
Texture2DMS<float4> SpecularShininess : register(t4);
Texture2DMS<float4> AmbientMetallic : register(t5);
Texture2DMS<float4> EmissionColor : register(t6);
Texture2DMS<float4> SampleVelocity : register(t7);

float4 VSMain(uint vertexId : SV_VertexID) : SV_POSITION
{
    float2 uv = float2((vertexId << 1u) & 2u, vertexId & 2u);
    return float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
}

uint NearestSample(int2 pixel, out float depth)
{
    uint width, height, count;
    SampleDepth.GetDimensions(width, height, count);
    uint nearest = 0u;
    depth = SampleDepth.Load(pixel, nearest);
    [loop] for (uint sampleIndex = 1u; sampleIndex < count; ++sampleIndex)
    {
        float candidate = SampleDepth.Load(pixel, sampleIndex);
        if (candidate < depth) { depth = candidate; nearest = sampleIndex; }
    }
    return nearest;
}

float PSDepth(float4 position : SV_POSITION) : SV_Depth
{
    float depth;
    NearestSample(int2(position.xy), depth);
    return depth;
}

struct MaterialOutput
{
    float4 Diffuse : SV_TARGET0;
    float4 Normal : SV_TARGET1;
    float4 Specular : SV_TARGET2;
    float4 Ambient : SV_TARGET3;
    float4 Emission : SV_TARGET4;
    float Depth : SV_Depth;
};
MaterialOutput PSMaterial(float4 position : SV_POSITION)
{
    MaterialOutput output;
    int2 pixel = int2(position.xy);
    uint nearest = NearestSample(pixel, output.Depth);
    output.Diffuse = DiffuseOpacity.Load(pixel, nearest);
    output.Normal = NormalRoughness.Load(pixel, nearest);
    output.Specular = SpecularShininess.Load(pixel, nearest);
    output.Ambient = AmbientMetallic.Load(pixel, nearest);
    output.Emission = EmissionColor.Load(pixel, nearest);
    return output;
}

float4 PSVelocity(float4 position : SV_POSITION) : SV_TARGET
{
    float depth;
    int2 pixel = int2(position.xy);
    uint nearest = NearestSample(pixel, depth);
    return SampleVelocity.Load(pixel, nearest);
}

float4 PSRestoreColor(float4 position : SV_POSITION) : SV_TARGET
{
    // Broadcast opaque post-processing to all samples before transparent draws.
    // The original per-sample opaque depth remains intact. At mixed-coverage
    // edges this approximates the filtered background with one pixel color.
    return ResolvedColor.Load(int3(int2(position.xy), 0));
}
