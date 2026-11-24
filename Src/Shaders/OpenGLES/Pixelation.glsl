#version 300 es
// Sample the center of each screen-aligned pixel block in linear light.
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
uniform vec4 NprTone;
out vec4 OutColor;
void main()
{
    vec2 extent = 1.0 / InverseExtentStrengths.xy;
    float block = max(1.0, floor(NprTone.y + 0.5));
    vec2 pixel = floor(gl_FragCoord.xy / block) * block + block * 0.5;
    vec2 uv = clamp(pixel, vec2(0.5), extent - 0.5) * InverseExtentStrengths.xy;
    OutColor = textureLod(SceneColor, uv, 0.0);
}
#endif
