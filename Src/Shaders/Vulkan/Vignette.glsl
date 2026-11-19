#version 450
// Exposure falloff in linear light. A normalized radial profile preserves
// the same corner strength across viewport sizes without touching alpha.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleFarCorner = 3.0;
    vec2 position = gl_VertexIndex == 0 ? vec2(-1.0, -1.0) :
        gl_VertexIndex == 1 ? vec2(TriangleFarCorner, -1.0) : vec2(-1.0, TriangleFarCorner);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform PostConstants
{
    vec4 InverseExtentStrengths;
    uvec4 GrainSeedReserved;
} Post;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
void main()
{
    const float FalloffStart = 0.2;
    const float FalloffEnd = 0.8;
    vec2 coordinate = gl_FragCoord.xy * Post.InverseExtentStrengths.xy;
    vec2 centered = coordinate * 2.0 - 1.0;
    float radiusSquared = dot(centered, centered) * 0.5;
    float exposure = 1.0 - Post.InverseExtentStrengths.z * smoothstep(FalloffStart, FalloffEnd, radiusSquared);
    vec4 color = texture(sampler2D(SceneColor, ColorSampler), coordinate);
    OutputColor = vec4(color.rgb * exposure, color.a);
}
#endif
