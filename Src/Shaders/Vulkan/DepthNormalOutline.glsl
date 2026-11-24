#version 450
// Fullscreen depth and world-normal discontinuities form geometric contours.
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
    vec4 DepthNormalEdge;
} Post;
layout(set = 0, binding = 1) uniform texture2D SceneDepth;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 3) uniform texture2D SceneNormal;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
void main()
{
    ivec2 colorExtent = textureSize(sampler2D(SceneColor, ColorSampler), 0);
    ivec2 extent = textureSize(sampler2D(SceneDepth, ColorSampler), 0);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy * vec2(extent) / vec2(colorExtent)), ivec2(0), extent - 1);
    vec4 color = texelFetch(sampler2D(SceneColor, ColorSampler), ivec2(gl_FragCoord.xy), 0);
    float depth = texelFetch(sampler2D(SceneDepth, ColorSampler), pixel, 0).r;
    vec3 normal = normalize(texelFetch(sampler2D(SceneNormal, ColorSampler), pixel, 0).xyz + vec3(1.0e-6));
    float depthEdge = 0.0, normalEdge = 0.0;
    const ivec2 offsets[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    for (int i = 0; i < 4; ++i)
    {
        ivec2 neighbor = clamp(pixel + offsets[i], ivec2(0), extent - 1);
        float nextDepth = texelFetch(sampler2D(SceneDepth, ColorSampler), neighbor, 0).r;
        depthEdge = max(depthEdge, abs(nextDepth - depth) / max(1.0 - max(nextDepth, depth), 0.001));
        vec3 nextNormal = normalize(texelFetch(sampler2D(SceneNormal, ColorSampler), neighbor, 0).xyz + vec3(1.0e-6));
        if (depth < 0.999999 && nextDepth < 0.999999)
            normalEdge = max(normalEdge, 1.0 - dot(normal, nextNormal));
    }
    float edge = max(smoothstep(Post.DepthNormalEdge.x, Post.DepthNormalEdge.x * 2.0, depthEdge),
        smoothstep(Post.DepthNormalEdge.y, Post.DepthNormalEdge.y + 0.1, normalEdge));
    OutputColor = vec4(color.rgb * (1.0 - edge * Post.DepthNormalEdge.z), color.a);
}
#endif
