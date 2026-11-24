#version 300 es
// Sobel color gradients draw a screen-space contour while preserving alpha.
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
uniform vec4 NprTone;
out vec4 OutColor;
vec3 at(ivec2 pixel, ivec2 extent)
{
    return texelFetch(SceneColor, clamp(pixel, ivec2(0), extent - 1), 0).rgb;
}
void main()
{
    ivec2 extent = textureSize(SceneColor, 0), pixel = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(SceneColor, pixel, 0);
    vec3 tl = at(pixel + ivec2(-1, -1), extent), tc = at(pixel + ivec2(0, -1), extent);
    vec3 tr = at(pixel + ivec2(1, -1), extent), ml = at(pixel + ivec2(-1, 0), extent);
    vec3 mr = at(pixel + ivec2(1, 0), extent), bl = at(pixel + ivec2(-1, 1), extent);
    vec3 bc = at(pixel + ivec2(0, 1), extent), br = at(pixel + ivec2(1, 1), extent);
    vec3 gx = tr + 2.0 * mr + br - tl - 2.0 * ml - bl;
    vec3 gy = bl + 2.0 * bc + br - tl - 2.0 * tc - tr;
    float edge = length(vec2(length(gx), length(gy))) * 0.25;
    float ink = smoothstep(NprTone.z, NprTone.z + 0.1, edge) * NprTone.w;
    OutColor = vec4(center.rgb * (1.0 - ink), center.a);
}
#endif
