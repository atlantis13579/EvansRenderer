#version 300 es
// Four-quadrant Kuwahara filtering retains low-variance color regions.
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
uniform vec4 Painterly;
out vec4 OutColor;
vec3 at(ivec2 pixel, ivec2 extent)
{
    return texelFetch(SceneColor, clamp(pixel, ivec2(0), extent - 1), 0).rgb;
}
void main()
{
    ivec2 extent = textureSize(SceneColor, 0), pixel = ivec2(gl_FragCoord.xy);
    vec4 center = texelFetch(SceneColor, pixel, 0);
    int radius = clamp(int(Painterly.x + 0.5), 1, 4);
    float bestVariance = 1.0e20;
    vec3 bestColor = center.rgb;
    for (int quadrant = 0; quadrant < 4; ++quadrant)
    {
        int xStart = (quadrant & 1) != 0 ? 0 : -radius;
        int yStart = (quadrant & 2) != 0 ? 0 : -radius;
        vec3 sum = vec3(0), sumSquared = vec3(0);
        for (int y = 0; y <= radius; ++y)
        for (int x = 0; x <= radius; ++x)
        {
            vec3 value = at(pixel + ivec2(xStart + x, yStart + y), extent);
            sum += value;
            sumSquared += value * value;
        }
        float count = float((radius + 1) * (radius + 1));
        vec3 mean = sum / count;
        float variance = dot(max(sumSquared / count - mean * mean, vec3(0)), vec3(0.2126, 0.7152, 0.0722));
        if (variance < bestVariance) { bestVariance = variance; bestColor = mean; }
    }
    OutColor = vec4(mix(center.rgb, bestColor, Painterly.y), center.a);
}
#endif
