#version 450
// Sobel color gradients draw a screen-space contour while preserving alpha.
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
vec3 at(ivec2 pixel, ivec2 extent)
{
    return texelFetch(sampler2D(SceneColor, ColorSampler), clamp(pixel, ivec2(0), extent - 1), 0).rgb;
}
void main()
{
    ivec2 extent = textureSize(sampler2D(SceneColor, ColorSampler), 0), pixel = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(sampler2D(SceneColor, ColorSampler), pixel, 0);
    vec3 tl = at(pixel + ivec2(-1, -1), extent), tc = at(pixel + ivec2(0, -1), extent);
    vec3 tr = at(pixel + ivec2(1, -1), extent), ml = at(pixel + ivec2(-1, 0), extent);
    vec3 mr = at(pixel + ivec2(1, 0), extent), bl = at(pixel + ivec2(-1, 1), extent);
    vec3 bc = at(pixel + ivec2(0, 1), extent), br = at(pixel + ivec2(1, 1), extent);
    vec3 gx = tr + 2.0 * mr + br - tl - 2.0 * ml - bl;
    vec3 gy = bl + 2.0 * bc + br - tl - 2.0 * tc - tr;
    float edge = length(vec2(length(gx), length(gy))) * 0.25;
    float ink = smoothstep(Post.NprTone.z, Post.NprTone.z + 0.1, edge) * Post.NprTone.w;
    OutputColor = vec4(center.rgb * (1.0 - ink), center.a);
}
#endif
