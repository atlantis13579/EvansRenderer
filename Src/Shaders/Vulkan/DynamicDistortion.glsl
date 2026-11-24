#version 450
// Animate two orthogonal screen waves; amplitudes are measured in pixels.
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
} Post;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
void main()
{
    const float Tau = 6.28318530718;
    vec2 uv = gl_FragCoord.xy * Post.InverseExtentStrengths.xy;
    float t = Post.DynamicWarp.w * Post.DynamicWarp.z;
    vec2 wave = vec2(sin((uv.y * Post.DynamicWarp.y + t) * Tau),
        sin((uv.x * Post.DynamicWarp.y * 0.7 - t * 0.6) * Tau));
    vec2 donor = uv + wave * Post.DynamicWarp.x * Post.InverseExtentStrengths.xy * vec2(1.0, 0.5);
    OutputColor = textureLod(sampler2D(SceneColor, ColorSampler), donor, 0.0);
}
#endif
