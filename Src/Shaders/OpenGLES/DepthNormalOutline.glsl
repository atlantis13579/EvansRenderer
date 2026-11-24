#version 300 es
// Fullscreen depth and world-normal discontinuities form geometric contours.
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
uniform sampler2D SceneDepth;
uniform sampler2D SceneNormal;
uniform vec4 DepthNormalEdge;
out vec4 OutColor;
void main()
{
    ivec2 colorExtent = textureSize(SceneColor, 0);
    ivec2 extent = textureSize(SceneDepth, 0);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy * vec2(extent) / vec2(colorExtent)), ivec2(0), extent - 1);
    vec4 color = texelFetch(SceneColor, ivec2(gl_FragCoord.xy), 0);
    float depth = texelFetch(SceneDepth, pixel, 0).r;
    vec3 normal = normalize(texelFetch(SceneNormal, pixel, 0).xyz + vec3(1.0e-6));
    float depthEdge = 0.0, normalEdge = 0.0;
    const ivec2 offsets[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    for (int i = 0; i < 4; ++i)
    {
        ivec2 neighbor = clamp(pixel + offsets[i], ivec2(0), extent - 1);
        float nextDepth = texelFetch(SceneDepth, neighbor, 0).r;
        depthEdge = max(depthEdge, abs(nextDepth - depth) / max(1.0 - max(nextDepth, depth), 0.001));
        vec3 nextNormal = normalize(texelFetch(SceneNormal, neighbor, 0).xyz + vec3(1.0e-6));
        if (depth < 0.999999 && nextDepth < 0.999999)
            normalEdge = max(normalEdge, 1.0 - dot(normal, nextNormal));
    }
    float edge = max(smoothstep(DepthNormalEdge.x, DepthNormalEdge.x * 2.0, depthEdge),
        smoothstep(DepthNormalEdge.y, DepthNormalEdge.y + 0.1, normalEdge));
    OutColor = vec4(color.rgb * (1.0 - edge * DepthNormalEdge.z), color.a);
}
#endif
