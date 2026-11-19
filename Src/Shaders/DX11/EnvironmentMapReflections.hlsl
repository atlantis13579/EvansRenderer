// DX11 latitude-longitude conversion, reflection lookup and RGB composition.
// PBRT (3rd ed.), "Infinite Area Lights": spherical directions parameterize a
// 2:1 image. Here +Y is north and the longitude seam faces -X. Conversion samples
// completed scene cube faces; lookup reads a separate mipmapped 2D image.
// Ordinary image mips approximate roughness, with nonuniform angular footprints
// near the poles. They do not integrate a GGX lobe or diffuse global illumination.
cbuffer EnvironmentMapSettings : register(b0)
{
    row_major float4x4 WorldFromClip;
    float4 TargetExtent;
    float4 ReflectionScale; // intensity, last image mip, reserved
    uint4 Operation; // phase, material model, reserved
    float4 ReceiverPlane, ReceiverMinimum, ReceiverMaximum;
};
Texture2D<float4> OpaqueRadiance : register(t0);
Texture2D<float> OpaqueDepth : register(t1);
Texture2D<float4> SurfaceNormal : register(t2);
Texture2D<float4> SurfaceSpecular : register(t3);
Texture2D<float4> SurfaceDiffuse : register(t4);
Texture2D<float4> SurfaceAmbient : register(t5);
Texture2D<float4> ReflectedRadiance : register(t6);
TextureCube<float4> CaptureCube : register(t7);
Texture2D<float4> LatitudeLongitude : register(t8);
SamplerState CaptureFilter : register(s0);
SamplerState LongitudeWrapLatitudeClamp : register(s1);

static const float Pi = 3.14159265358979323846f;
static const float TwoPi = 2 * Pi;
static const float HalfTurn = 0.5f;
static const float MinimumVectorSquared = 1e-12f;
static const float HomogeneousTolerance = 1e-7f;
static const float InteriorClipDepth = 0.5f;
static const float ClearDepth = 1.0f;
static const float DielectricReflectance = 0.04f;
static const float SchlickPower = 5.0f;
static const float MaximumHalfRadiance = 65504.0f;
static const float HalfRoundingMargin = 32.0f;
static const uint CompositeOperation = 1u;
static const uint MetallicRoughnessModel = 3u;

struct ImageVertex { float4 Position : SV_POSITION; };
ImageVertex VSMain(uint vertex : SV_VertexID)
{
    const float OversizedCorner = 3.0f;
    float2 xy = vertex == 0 ? float2(-1, -1) :
        vertex == 1 ? float2(-1, OversizedCorner) : float2(OversizedCorner, -1);
    ImageVertex output;
    output.Position = float4(xy, 0, 1);
    return output;
}

float4 PSConvert(ImageVertex input) : SV_TARGET
{
    float2 uv = input.Position.xy * TargetExtent.zw;
    float longitude = (uv.x - HalfTurn) * TwoPi;
    float colatitude = uv.y * Pi;
    float3 direction = float3(sin(colatitude) * cos(longitude), cos(colatitude), sin(colatitude) * sin(longitude));
    // Shared capture cameras are right-handed; native cube Z has the opposite sign.
    float3 nativeDirection = float3(direction.xy, -direction.z);
    return float4(max(CaptureCube.SampleLevel(CaptureFilter, nativeDirection, 0).rgb, 0), 1);
}

bool UnitDirection(inout float3 direction)
{
    float squared = dot(direction, direction);
    if (!all(isfinite(direction)) || squared <= MinimumVectorSquared) return false;
    direction *= rsqrt(squared);
    return true;
}
bool Unproject(float2 screen, float depth, out float3 worldPosition)
{
    float2 uv = screen * TargetExtent.zw;
    float4 h = mul(WorldFromClip, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    worldPosition = 0;
    if (abs(h.w) <= HomogeneousTolerance) return false;
    worldPosition = h.xyz / h.w;
    return all(isfinite(worldPosition));
}

float4 PSMain(ImageVertex input) : SV_TARGET
{
    int2 pixel = int2(input.Position.xy);
    if (Operation.x == CompositeOperation) return float4(ReflectedRadiance.Load(int3(pixel, 0)).rgb, 0);
    float4 surface = SurfaceNormal.Load(int3(pixel, 0));
    float3 normal = surface.xyz;
    if (ReflectionScale.x <= 0 || OpaqueDepth.Load(int3(pixel, 0)) >= ClearDepth || surface.a < 0 || !UnitDirection(normal)) return 0;
    float3 nearPoint, interiorPoint;
    if (Operation.w != 0u)
    {
        float3 receiver;
        bool onMirror = Unproject(input.Position.xy, OpaqueDepth.Load(int3(pixel, 0)), receiver) &&
            abs(dot(ReceiverPlane, float4(receiver, 1))) <= ReceiverMinimum.w &&
            abs(dot(normal, ReceiverPlane.xyz)) >= ReceiverMaximum.w;
        const uint BoundedReceiver = 2u;
        bool inside = Operation.w != BoundedReceiver ||
            (all(receiver >= ReceiverMinimum.xyz) && all(receiver <= ReceiverMaximum.xyz));
        if (onMirror && inside) return 0;
    }
    if (!Unproject(input.Position.xy, 0, nearPoint) || !Unproject(input.Position.xy, InteriorClipDepth, interiorPoint)) return 0;
    float3 incident = interiorPoint - nearPoint;
    if (!UnitDirection(incident)) return 0;
    float3 direction = reflect(incident, normal);
    // atan2(0,0) has no defined longitude. Every longitude meets at a pole.
    float longitude = dot(direction.xz, direction.xz) > MinimumVectorSquared ? atan2(direction.z, direction.x) : 0;
    float2 uv = float2(longitude / TwoPi + HalfTurn, acos(clamp(direction.y, -1, 1)) / Pi);
    float mip = saturate(surface.a) * ReflectionScale.y;
    float3 incoming = max(LatitudeLongitude.SampleLevel(LongitudeWrapLatitudeClamp, uv, mip).rgb, 0);
    float3 f0 = saturate(SurfaceSpecular.Load(int3(pixel, 0)).rgb);
    if (Operation.y == MetallicRoughnessModel)
        f0 = lerp(f0, saturate(SurfaceDiffuse.Load(int3(pixel, 0)).rgb),
            saturate(SurfaceAmbient.Load(int3(pixel, 0)).a));
    float cosine = saturate(dot(normal, -incident));
    // Analytic split-sum BRDF fit: Karis, Physically Based Shading on Mobile (2014).
    // https://www.unrealengine.com/blog/physically-based-shading-on-mobile
    // Rough diffuse surfaces must not become mirrors at grazing incidence.
    float4 brdfFit=clamp(surface.a,0.0,1.0)*float4(-1.0,-0.0275,-0.572,0.022)+float4(1.0,0.0425,1.04,-0.04);
    float grazing=min(brdfFit.x*brdfFit.x,exp2(-9.28*cosine))*brdfFit.x+brdfFit.y;
    float2 integrated=float2(-1.04,1.04)*grazing+brdfFit.zw;
    float3 fresnel = Operation.y == MetallicRoughnessModel?clamp(f0*integrated.x+integrated.y,0.0,1.0):f0 + (1 - f0) * pow(1 - cosine, SchlickPower);
    float3 headroom = max(MaximumHalfRadiance - HalfRoundingMargin - OpaqueRadiance.Load(int3(pixel, 0)).rgb, 0);
    return float4(min(incoming * fresnel * ReflectionScale.x, headroom), 0);
}
