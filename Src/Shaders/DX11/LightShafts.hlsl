// Native DX11 depth-occluded radial scattering; no shader-source includes.
// Mitchell, "Volumetric Light Scattering as a Post-Process", GPU Gems 3 (2007).
// A compact analytic source disk seeds a midpoint integral toward its center.
// Only visible screen-space occluders participate; transparent surfaces follow.
cbuffer ShaftParameters : register(b0)
{
    float4 Source;       // top-left UV, depth, radius in image heights
    float4 Radiance;     // RGB source radiance, scattering intensity
    float4 Integration;  // decay, depth bias, width, height
    uint4 Control;       // sample count, mask/composite stage, visible source
};
Texture2D<float4> Scene : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> SourceMask : register(t2);
static const uint MaskStage = 0;
static const float HalfFloatMaximum = 65504.0;
struct ScreenVertex { float4 Position : SV_Position; };
ScreenVertex VSMain(uint index : SV_VertexID)
{
    const float FullscreenEdge = 3.0;
    ScreenVertex output;
    output.Position = float4(index == 1 ? float2(FullscreenEdge, -1.0) :
        index == 2 ? float2(-1.0, FullscreenEdge) : float2(-1.0, -1.0), 0.0, 1.0);
    return output;
}
float MaskTexel(int2 pixel)
{
    if (any(pixel < 0) || any(pixel >= int2(Integration.zw))) return 0.0;
    return SourceMask.Load(int3(pixel, 0)).r;
}
float MaskLinear(float2 uv)
{
    // Zero-border bilinear reconstruction prevents bright edge replication.
    float2 location = uv * Integration.zw - 0.5;
    int2 corner = int2(floor(location));
    float2 weight = frac(location);
    return lerp(lerp(MaskTexel(corner), MaskTexel(corner + int2(1, 0)), weight.x),
        lerp(MaskTexel(corner + int2(0, 1)), MaskTexel(corner + int2(1, 1)), weight.x), weight.y);
}
float4 PSMain(ScreenVertex input) : SV_Target0
{
    int2 pixel = int2(input.Position.xy);
    float2 uv = input.Position.xy / Integration.zw;
    if (Control.y == MaskStage)
    {
        float visibility = 0.0;
        if (Control.z != 0 && SceneDepth.Load(int3(pixel, 0)) + Integration.y >= Source.z)
        {
            float2 delta = (uv - Source.xy) * float2(Integration.z / Integration.w, 1.0);
            float profile = saturate(1.0 - length(delta) / Source.w);
            visibility = profile * profile;
        }
        return float4(visibility, 0.0, 0.0, 1.0);
    }
    float4 scene = Scene.Load(int3(pixel, 0));
    if (Control.z == 0) return scene;
    float scattering = 0.0;
    [loop] for (uint sampleIndex = 0; sampleIndex < Control.x; ++sampleIndex)
    {
        float t = (float(sampleIndex) + 0.5) / float(Control.x);
        scattering += MaskLinear(lerp(uv, Source.xy, t)) * exp(-Integration.x * t);
    }
    scattering /= float(Control.x);
    return float4(min(scene.rgb + Radiance.rgb * (Radiance.a * scattering), HalfFloatMaximum), scene.a);
}
