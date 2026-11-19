#version 300 es
// Isotropic XDoG and analytic tonal hatching, authored for OpenGL ES.
// Winnemoeller et al. (2012): https://www.kyprianidis.com/p/cag2012/
// No flow field or external paper textures; hatch coordinates are screen-fixed.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FarCorner = 3.0;
    vec2 corner = gl_VertexID == 0 ? vec2(-1, -1) : gl_VertexID == 1 ? vec2(FarCorner, -1) : vec2(-1, FarCorner);
    gl_Position = vec4(corner, 0, 1);
}
#else
uniform sampler2D SceneColor;
uniform vec4 InverseExtentStrengths;
uniform vec4 NprEdge;
uniform vec4 NprStyle;
uniform vec4 NprHatch;
uniform vec4 NprTone;
out vec4 OutColor;
vec3 encodeDisplay(vec3 linear)
{
    const float LinearThreshold = 0.0031308, LinearScale = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, PowerExponent = 2.4;
    return mix(PowerScale * pow(clamp(linear, 0.0, 1.0), vec3(1.0 / PowerExponent)) - PowerOffset,
        linear * LinearScale, lessThanEqual(linear, vec3(LinearThreshold)));
}
vec3 decodeDisplay(vec3 display)
{
    const float DisplayThreshold = 0.04045, LinearScale = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, PowerExponent = 2.4;
    return mix(pow((display + PowerOffset) / PowerScale, vec3(PowerExponent)),
        display / LinearScale, lessThanEqual(display, vec3(DisplayThreshold)));
}
float contourAt(ivec2 center)
{
    const float GaussianRatio = 1.6, SupportSigma = 3.0;
    const int MaximumRadius = 8; // ceil(3 * 1.6 * maximum sigma 1.5).
    const vec3 LumaWeights = vec3(0.2126, 0.7152, 0.0722);
    vec2 sigma = NprEdge.x * vec2(1.0, GaussianRatio);
    int radius = min(MaximumRadius, int(ceil(SupportSigma * sigma.y)));
    ivec2 lastPixel = textureSize(SceneColor, 0) - 1;
    vec2 weighted = vec2(0), normalization = vec2(0);
    for (int row = -radius; row <= radius; ++row)
        for (int column = -radius; column <= radius; ++column)
        {
            ivec2 donor = clamp(center + ivec2(column, row), ivec2(0), lastPixel);
            float luma = dot(encodeDisplay(texelFetch(SceneColor, donor, 0).rgb), LumaWeights);
            vec2 weights = exp(-float(column * column + row * row) / (2.0 * sigma * sigma));
            weighted += weights * luma;
            normalization += weights;
        }
    vec2 filtered = weighted / normalization;
    float difference = filtered.x - NprEdge.y * filtered.y;
    float e = exp(2.0 * min(0.0, NprEdge.w * (difference - NprEdge.z)));
    return 2.0 * e / (1.0 + e); // Stable form of the XDoG white/tanh threshold.
}
float hatchAt(vec2 imagePixel, vec2 normal)
{
    float distanceToLine = abs(fract(dot(imagePixel, normal) / NprHatch.x + 0.5) - 0.5) * NprHatch.x;
    float halfWidth = NprHatch.y * 0.5;
    float halfPixel = (abs(normal.x) + abs(normal.y)) * 0.5;
    return 1.0 - smoothstep(max(0.0, halfWidth - halfPixel), halfWidth + halfPixel, distanceToLine);
}
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 source = texelFetch(SceneColor, pixel, 0);
    vec3 display = clamp(encodeDisplay(source.rgb), 0.0, 1.0);
    // Only hatch coordinates flip: GLES texel addresses retain the native origin.
    vec2 imagePixel = vec2(gl_FragCoord.x, float(textureSize(SceneColor, 0).y) - gl_FragCoord.y);
    const vec3 LumaWeights = vec3(0.2126, 0.7152, 0.0722);
    const float FirstTone = 0.25, MiddleTone = 0.5, DarkTone = 0.75, ToneFade = 0.1;
    const float DiagonalUnitScale = 0.7071067811865475;
    vec3 thresholds = vec3(FirstTone, MiddleTone, DarkTone);
    vec3 masks = smoothstep(thresholds - ToneFade, thresholds + ToneFade, vec3(1.0 - dot(display, LumaWeights)));
    vec2 normal = NprHatch.zw;
    vec3 coverage = vec3(hatchAt(imagePixel, normal), hatchAt(imagePixel, vec2(-normal.y, normal.x)),
        hatchAt(imagePixel, vec2(normal.x - normal.y, normal.x + normal.y) * DiagonalUnitScale));
    vec3 transmission = 1.0 - coverage * masks * NprStyle.y;
    float edge = NprStyle.x > 0.0 ? contourAt(pixel) : 1.0;
    float intervals = NprTone.x - 1.0;
    vec3 posterized = floor(display * intervals + 0.5) / intervals;
    vec3 paper = mix(vec3(1), posterized, NprStyle.z);
    vec3 stylized = paper * mix(1.0, edge, NprStyle.x) * transmission.x * transmission.y * transmission.z;
    // The sRGB attachment performs the final encode. Alpha stays linear.
    OutColor = vec4(decodeDisplay(mix(display, stylized, NprStyle.w)), source.a);
}
#endif
