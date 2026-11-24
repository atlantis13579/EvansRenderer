#version 300 es
// Animate two orthogonal screen waves; amplitudes are measured in pixels.
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
uniform vec4 InverseExtentStrengths;
uniform vec4 DynamicWarp;
out vec4 OutColor;
void main()
{
    const float Tau = 6.28318530718;
    vec2 uv = gl_FragCoord.xy * InverseExtentStrengths.xy;
    float t = DynamicWarp.w * DynamicWarp.z;
    vec2 wave = vec2(sin((uv.y * DynamicWarp.y + t) * Tau),
        sin((uv.x * DynamicWarp.y * 0.7 - t * 0.6) * Tau));
    vec2 donor = uv + wave * DynamicWarp.x * InverseExtentStrengths.xy * vec2(1.0, 0.5);
    OutColor = textureLod(SceneColor, donor, 0.0);
}
#endif
