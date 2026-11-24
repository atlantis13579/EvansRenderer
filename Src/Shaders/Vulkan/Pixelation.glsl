#version 450
// Sample the center of each screen-aligned pixel block in linear light.
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
} Post;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
void main()
{
    vec2 extent = 1.0 / Post.InverseExtentStrengths.xy;
    float block = max(1.0, floor(Post.NprTone.y + 0.5));
    vec2 pixel = floor(gl_FragCoord.xy / block) * block + block * 0.5;
    vec2 uv = clamp(pixel, vec2(0.5), extent - 0.5) * Post.InverseExtentStrengths.xy;
    OutputColor = textureLod(sampler2D(SceneColor, ColorSampler), uv, 0.0);
}
#endif
