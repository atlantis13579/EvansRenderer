#version 300 es
// Bilateral pigment diffusion, tonal quantization and static paper granulation.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
        gl_VertexID == 1 ? vec2(3.0, -1.0) : vec2(-1.0, 3.0);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
uniform sampler2D SceneColor;
uniform vec4 Painterly;
out vec4 OutColor;
vec3 at(ivec2 pixel, ivec2 extent)
{
    return texelFetch(SceneColor, clamp(pixel, ivec2(0), extent - 1), 0).rgb;
}
void main()
{
    ivec2 extent = textureSize(SceneColor, 0), pixel = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(SceneColor, pixel, 0);
    int radius = clamp(int(Painterly.z + 0.5), 1, 4);
    float sigma = max(float(radius) * 0.7, 0.7);
    vec3 sum = vec3(0);
    float weightSum = 0.0;
    for (int y = -radius; y <= radius; ++y)
    for (int x = -radius; x <= radius; ++x)
    {
        vec3 sampleColor = at(pixel + ivec2(x, y), extent);
        vec3 difference = sampleColor - center.rgb;
        float weight = exp(-float(x * x + y * y) / (2.0 * sigma * sigma) - dot(difference, difference) * 8.0);
        sum += sampleColor * weight;
        weightSum += weight;
    }
    vec3 pigment = sum / max(weightSum, 1.0e-6);
    pigment = floor(pigment * 12.0 + 0.5) / 12.0;
    float paper = fract(sin(dot(vec2(pixel), vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    pigment = clamp(pigment + paper * 0.025, 0.0, 1.0);
    OutColor = vec4(mix(center.rgb, pigment, Painterly.w), center.a);
}
#endif
