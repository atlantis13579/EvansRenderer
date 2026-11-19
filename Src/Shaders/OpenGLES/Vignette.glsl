#version 300 es
// A smooth radial exposure falloff multiplies linear RGB and preserves alpha.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float OversizedCorner = 3.0;
    vec2 position = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
        gl_VertexID == 1 ? vec2(OversizedCorner, -1.0) : vec2(-1.0, OversizedCorner);
    gl_Position = vec4(position, 0.0, 1.0);
}
#else
uniform sampler2D SceneColor;
uniform vec4 InverseExtentStrengths;
out vec4 OutColor;
void main()
{
    const float InnerRadiusSquared = 0.2;
    const float OuterRadiusSquared = 0.8;
    vec2 uv = gl_FragCoord.xy * InverseExtentStrengths.xy;
    vec2 centered = uv * 2.0 - 1.0;
    float radial = dot(centered, centered) * 0.5;
    float attenuation = 1.0 - InverseExtentStrengths.z * smoothstep(InnerRadiusSquared, OuterRadiusSquared, radial);
    vec4 color = texture(SceneColor, uv);
    OutColor = vec4(color.rgb * attenuation, color.a);
}
#endif
