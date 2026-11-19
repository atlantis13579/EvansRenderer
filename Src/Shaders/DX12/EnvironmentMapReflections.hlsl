// Independent DX12 latitude-longitude technique. Spherical coordinates store
// directional radiance in a 2:1 HDR image (PBRT, 3rd ed., Infinite Area Lights).
// A completed scene cube supplies the base image; a separate render/copy chain
// creates each smaller image mip. Lookup wraps longitude at -X and clamps poles.
// This is a local specular probe approximation, without diffuse GI or GGX
// prefiltering. Ordinary box mips have unequal angular footprints near the poles.
cbuffer EnvironmentParameters : register(b0)
{
    row_major float4x4 ClipToWorld;
    float4 RasterExtent; // destination width/height and reciprocals
    float4 ReflectionParameters; // intensity, highest image mip, reserved
    uint4 PhaseAndModel; // lookup/composite operation, surface model, reserved
    float4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
};
Texture2D<float4> SceneRadiance : register(t0);
Texture2D<float> SceneDepth : register(t1);
Texture2D<float4> NormalAndRoughness : register(t2);
Texture2D<float4> SpecularAndShininess : register(t3);
Texture2D<float4> DiffuseAndOpacity : register(t4);
Texture2D<float4> AmbientAndMetallic : register(t5);
Texture2D<float4> ReflectionImage : register(t6);
TextureCube<float4> CapturedFaces : register(t7);
Texture2D<float4> EnvironmentImage : register(t9);
Texture2D<float4> ReductionSource : register(t10); // Persistent SRV of one parent mip.
SamplerState CubeSampling : register(s0);
SamplerState SphericalSampling : register(s1); // Repeat U, clamp V/W, trilinear.

static const float Pi = 3.14159265358979323846f;
static const float FullTurn = 2 * Pi;
static const float LongitudeCenter = 0.5f;
static const uint FilterSpan = 2u;
static const float BoxWeight = 1.0f / (FilterSpan * FilterSpan);
static const uint CompositePhase = 1u;
static const uint PbrModel = 3u;
static const float DepthClearValue = 1.0f;
static const float FiniteRayDepth = 0.5f;
static const float HomogeneousEpsilon = 1e-7f;
static const float DirectionSquaredEpsilon = 1e-12f;
static const float DielectricF0 = 0.04f;
static const float FresnelPower = 5.0f;
static const float HalfRadianceLimit = 65504.0f;
static const float HalfRoundingGuard = 32.0f;

struct TriangleOutput { float4 PixelPosition : SV_POSITION; };
TriangleOutput VSMain(uint index : SV_VertexID)
{
    float2 corner = float2(index & 2u, (index << 1u) & 2u);
    TriangleOutput output;
    output.PixelPosition = float4(corner * 2 - 1, 0, 1);
    return output;
}

float4 PSConvert(TriangleOutput input) : SV_TARGET
{
    float2 address = input.PixelPosition.xy * RasterExtent.zw;
    float longitude = (address.x - LongitudeCenter) * FullTurn;
    float colatitude = address.y * Pi;
    float longitudeSin, longitudeCos, latitudeSin, latitudeCos;
    sincos(longitude, longitudeSin, longitudeCos);
    sincos(colatitude, latitudeSin, latitudeCos);
    // Convert Y-up world directions to native cube axes only at this boundary.
    float3 nativeDirection = float3(latitudeSin * longitudeCos, latitudeCos, -latitudeSin * longitudeSin);
    return float4(max(CapturedFaces.SampleLevel(CubeSampling, nativeDirection, 0).rgb, 0), 1);
}

float4 PSMip(TriangleOutput input) : SV_TARGET
{
    uint width, height;
    ReductionSource.GetDimensions(width, height);
    uint2 origin = uint2(input.PixelPosition.xy) * FilterSpan;
    uint2 lastTexel = uint2(width, height) - 1;
    float4 filtered = 0;
    for (uint y = 0; y < FilterSpan; ++y)
        for (uint x = 0; x < FilterSpan; ++x)
            filtered += ReductionSource.Load(int3(min(origin + uint2(x, y), lastTexel), 0));
    // The final 2x1 -> 1x1 reduction repeats the sole row, preserving its average.
    return filtered * BoxWeight;
}

bool NormalizeFinite(inout float3 direction)
{
    float squaredLength = dot(direction, direction);
    if (!all(isfinite(direction)) || squaredLength <= DirectionSquaredEpsilon) return false;
    direction *= rsqrt(squaredLength);
    return true;
}
bool CameraPosition(float2 pixel, float depth, out float3 world)
{
    float2 uv = pixel * RasterExtent.zw;
    float4 h = mul(ClipToWorld, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    world = 0;
    if (abs(h.w) < HomogeneousEpsilon) return false;
    world = h.xyz / h.w;
    return all(isfinite(world));
}

float4 PSMain(TriangleOutput input) : SV_TARGET
{
    int3 pixel = int3(int2(input.PixelPosition.xy), 0);
    if (PhaseAndModel.x == CompositePhase) return float4(ReflectionImage.Load(pixel).rgb, 0);
    float4 surface = NormalAndRoughness.Load(pixel);
    float3 normal = surface.xyz;
    if (ReflectionParameters.x <= 0 || SceneDepth.Load(pixel) >= DepthClearValue || surface.a < 0 || !NormalizeFinite(normal)) return 0;
    float3 nearWorld, interiorWorld;
    if (PhaseAndModel.w != 0u)
    {
        float3 receiver;
        bool onPlane = CameraPosition(input.PixelPosition.xy, SceneDepth.Load(pixel), receiver) &&
            abs(dot(ReceiverPlane.xyz, receiver) + ReceiverPlane.w) <= ReceiverMinimum.w &&
            abs(dot(ReceiverPlane.xyz, normal)) >= ReceiverMaximum.w;
        const uint FiniteMirror = 2u;
        if (onPlane && (PhaseAndModel.w != FiniteMirror ||
            (all(receiver >= ReceiverMinimum.xyz) && all(receiver <= ReceiverMaximum.xyz)))) return 0;
    }
    if (!CameraPosition(input.PixelPosition.xy, 0, nearWorld) || !CameraPosition(input.PixelPosition.xy, FiniteRayDepth, interiorWorld)) return 0;
    // Two finite depth positions also define rays for orthographic/infinite-far views.
    float3 viewRay = interiorWorld - nearWorld;
    if (!NormalizeFinite(viewRay)) return 0;
    float3 ray = viewRay - 2 * dot(viewRay, normal) * normal;
    float longitude = dot(ray.xz, ray.xz) > DirectionSquaredEpsilon ? atan2(ray.z, ray.x) : 0;
    float2 environmentUV = float2(longitude / FullTurn + LongitudeCenter, acos(clamp(ray.y, -1, 1)) / Pi);
    float selectedMip = saturate(surface.a) * ReflectionParameters.y;
    float3 incoming = max(EnvironmentImage.SampleLevel(SphericalSampling, environmentUV, selectedMip).rgb, 0);

    float3 reflectance = saturate(SpecularAndShininess.Load(pixel).rgb);
    if (PhaseAndModel.y == PbrModel)
        reflectance = lerp(reflectance, saturate(DiffuseAndOpacity.Load(pixel).rgb), saturate(AmbientAndMetallic.Load(pixel).a));
    float cosine = saturate(dot(normal, -viewRay));
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    float4 brdfFit=clamp(surface.a,0.0,1.0)*float4(-1.0,-0.0275,-0.572,0.022)+float4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    float2 integrated=float2(-1.04,1.04)*grazing+brdfFit.zw;
    float3 fresnel = PhaseAndModel.y == PbrModel?clamp(reflectance*integrated.x+integrated.y,0.0,1.0):reflectance + (1 - reflectance) * pow(1 - cosine, FresnelPower);
    float3 remaining = max(HalfRadianceLimit - HalfRoundingGuard - SceneRadiance.Load(pixel).rgb, 0);
    return float4(min(ReflectionParameters.x * incoming * fresnel, remaining), 0);
}
