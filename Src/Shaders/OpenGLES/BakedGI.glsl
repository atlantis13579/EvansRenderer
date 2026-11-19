#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Independent ES probe bake. Sample-table and probe-atlas row zero both use
// native texel row zero; these data textures have no camera-image orientation.
// SH convolution: https://graphics.stanford.edu/papers/envmap/
// Visibility moments: https://jcgt.org/published/0008/02/01/
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
#else
uniform mat4 InverseViewProjection;
uniform vec4 GridMinimum, GridMaximum, Viewport, Appearance, Visibility;
uniform uvec4 Dimensions, Options;
uniform sampler2D RaySamples;
layout(location = 0) out vec4 OutColor;
const float Pi = 3.14159265358979323846, CoefficientScale = 0.25;
const float MaximumHalf = 65504.0, MinimumWeight = 1e-12, PixelCenter = 0.5;
const uint IrradiancePass = 0u;
const int MetadataBorder = 1;
float RealSH(int index, vec3 n)
{
    if (index == 0) return sqrt(1.0 / (4.0 * Pi));
    if (index == 1) return sqrt(3.0 / (4.0 * Pi)) * n.y;
    if (index == 2) return sqrt(3.0 / (4.0 * Pi)) * n.z;
    if (index == 3) return sqrt(3.0 / (4.0 * Pi)) * n.x;
    if (index == 4) return sqrt(15.0 / (4.0 * Pi)) * n.x * n.y;
    if (index == 5) return sqrt(15.0 / (4.0 * Pi)) * n.y * n.z;
    if (index == 6) return sqrt(5.0 / (16.0 * Pi)) * (3.0 * n.z * n.z - 1.0);
    if (index == 7) return sqrt(15.0 / (4.0 * Pi)) * n.x * n.z;
    return sqrt(15.0 / (16.0 * Pi)) * (n.x * n.x - n.y * n.y);
}
vec3 DecodeOct(vec2 uv)
{
    vec2 p = uv * 2.0 - 1.0;
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    float fold = clamp(-n.z, 0.0, 1.0);
    n.x += n.x >= 0.0 ? -fold : fold;
    n.y += n.y >= 0.0 ? -fold : fold;
    return normalize(n);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (Options.x == IrradiancePass)
    {
        vec3 result = vec3(0.0);
        for (uint ray = 0u; ray < Dimensions.w; ++ray)
        {
            int column = int(ray) + MetadataBorder;
            vec4 direction = texelFetch(RaySamples, ivec2(column, 0), 0);
            vec3 radiance = texelFetch(RaySamples, ivec2(column, pixel.y + MetadataBorder), 0).rgb;
            result += radiance * direction.w * RealSH(pixel.x, direction.xyz);
        }
        float band = pixel.x == 0 ? 1.0 : pixel.x < 4 ? 2.0 / 3.0 : 1.0 / 4.0;
        float probeActive = texelFetch(RaySamples, ivec2(0, pixel.y + MetadataBorder), 0).a;
        OutColor = vec4(clamp(result * (band * CoefficientScale), -MaximumHalf, MaximumHalf), probeActive);
        return;
    }
    int size = int(Options.z);
    int probe = pixel.x / size + (pixel.y / size) * int(Visibility.w);
    if (probe >= int(Options.w)) { OutColor = vec4(1.0, 1.0, 0.0, 0.0); return; }
    vec3 direction = DecodeOct((vec2(pixel % size) + PixelCenter) / float(size));
    vec2 moments = vec2(0.0); float total = 0.0;
    for (uint ray = 0u; ray < Dimensions.w; ++ray)
    {
        int column = int(ray) + MetadataBorder;
        vec3 omega = texelFetch(RaySamples, ivec2(column, 0), 0).xyz;
        float d = clamp(texelFetch(RaySamples, ivec2(column, probe + MetadataBorder), 0).w / Visibility.z, 0.0, 1.0);
        float weight = pow(clamp(dot(omega, direction), 0.0, 1.0), Visibility.y);
        moments += vec2(d, d*d) * weight; total += weight;
    }
    OutColor = vec4(total > MinimumWeight ? moments / total : vec2(1.0), 0.0, 1.0);
}
#endif
