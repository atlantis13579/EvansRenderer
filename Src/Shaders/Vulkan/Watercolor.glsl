#version 450
// Bilateral pigment diffusion, tonal quantization and static paper granulation.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = gl_VertexIndex == 0 ? vec2(-1.0, -1.0) :
        gl_VertexIndex == 1 ? vec2(3.0, -1.0) : vec2(-1.0, 3.0);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform PostConstants
{
    vec4 InverseExtentStrengths;
    uvec4 GrainSeedReserved;
    vec4 DistortionStrengthRadiusWeights;
    vec4 NprEdge;
    vec4 NprStyle;
    vec4 NprHatch;
    vec4 NprTone;
    vec4 DynamicWarp;
    vec4 Painterly;
} Post;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
vec3 at(ivec2 pixel, ivec2 extent)
{
    return texelFetch(sampler2D(SceneColor, ColorSampler), clamp(pixel, ivec2(0), extent - 1), 0).rgb;
}
void main()
{
    ivec2 extent = textureSize(sampler2D(SceneColor, ColorSampler), 0), pixel = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(sampler2D(SceneColor, ColorSampler), pixel, 0);
    int radius = clamp(int(Post.Painterly.z + 0.5), 1, 4);
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
    OutputColor = vec4(mix(center.rgb, pigment, Post.Painterly.w), center.a);
}
#endif
