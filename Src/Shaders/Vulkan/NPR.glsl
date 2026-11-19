#version 450
// Vulkan XDoG ink and procedural crosshatching. Compiled from GLSL at runtime.
// Reference: Winnemoeller et al. (2012), https://www.kyprianidis.com/p/cag2012/
// Filtering operates on display luma; contour strength, tonal layers and source
// color retention are independent controls in a single bounded image pass.
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleFarCorner = 3.0;
    vec2 position = gl_VertexIndex == 0 ? vec2(-1, -1) :
        gl_VertexIndex == 1 ? vec2(TriangleFarCorner, -1) : vec2(-1, TriangleFarCorner);
    gl_Position = vec4(position, 0, 1);
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
vec3 toDisplay(vec3 color)
{
    const float LinearThreshold = 0.0031308, LinearSlope = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, Exponent = 2.4;
    vec3 power = PowerScale * pow(clamp(color, 0.0, 1.0), vec3(1.0 / Exponent)) - PowerOffset;
    return mix(power, color * LinearSlope, lessThanEqual(color, vec3(LinearThreshold)));
}
vec3 toLinear(vec3 color)
{
    const float EncodedThreshold = 0.04045, LinearSlope = 12.92;
    const float PowerScale = 1.055, PowerOffset = 0.055, Exponent = 2.4;
    vec3 power = pow((color + PowerOffset) / PowerScale, vec3(Exponent));
    return mix(power, color / LinearSlope, lessThanEqual(color, vec3(EncodedThreshold)));
}
float xdog(ivec2 pixel, ivec2 extent)
{
    const float WideScale = 1.6, TruncationSigma = 3.0;
    const int RadiusLimit = 8; // Three-sigma support at the largest allowed scale.
    const vec3 DisplayLuma = vec3(0.2126, 0.7152, 0.0722);
    vec2 variance = Post.NprEdge.x * Post.NprEdge.x * vec2(1, WideScale * WideScale);
    int radius = min(RadiusLimit, int(ceil(TruncationSigma * Post.NprEdge.x * WideScale)));
    vec2 radiance = vec2(0), weights = vec2(0);
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
        {
            ivec2 address = clamp(pixel + ivec2(x, y), ivec2(0), extent - 1);
            vec3 linear = texelFetch(sampler2D(SceneColor, ColorSampler), address, 0).rgb;
            vec2 gaussian = exp(-float(x * x + y * y) / (2.0 * variance));
            radiance += gaussian * dot(toDisplay(linear), DisplayLuma);
            weights += gaussian;
        }
    vec2 blurred = radiance / weights;
    float difference = blurred.x - Post.NprEdge.y * blurred.y;
    // For negative arguments 1+tanh(x) = 2 exp(2x)/(1+exp(2x)); no overflow.
    float e = exp(2.0 * min(0.0, Post.NprEdge.w * (difference - Post.NprEdge.z)));
    return 2.0 * e / (1.0 + e);
}
float hatch(vec2 pixel, vec2 axis)
{
    float distance = abs(fract(dot(pixel, axis) / Post.NprHatch.x + 0.5) - 0.5) * Post.NprHatch.x;
    float halfWidth = Post.NprHatch.y * 0.5;
    float footprint = dot(abs(axis), vec2(1)) * 0.5;
    return 1.0 - smoothstep(max(0.0, halfWidth - footprint), halfWidth + footprint, distance);
}
void main()
{
    ivec2 extent = textureSize(sampler2D(SceneColor, ColorSampler), 0);
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 rgba = texelFetch(sampler2D(SceneColor, ColorSampler), pixel, 0);
    vec3 display = clamp(toDisplay(rgba.rgb), 0.0, 1.0);
    const vec3 DisplayLuma = vec3(0.2126, 0.7152, 0.0722);
    const float LightTone = 0.25, MiddleTone = 0.5, DarkTone = 0.75, Transition = 0.1;
    const float DiagonalScale = 0.7071067811865475;
    vec3 levels = vec3(LightTone, MiddleTone, DarkTone);
    vec3 tonalCoverage = smoothstep(levels - Transition, levels + Transition, vec3(1.0 - dot(display, DisplayLuma)));
    vec2 direction = Post.NprHatch.zw;
    vec3 strokes = vec3(hatch(gl_FragCoord.xy, direction),
        hatch(gl_FragCoord.xy, vec2(-direction.y, direction.x)),
        hatch(gl_FragCoord.xy, vec2(direction.x - direction.y, direction.x + direction.y) * DiagonalScale));
    vec3 transmission = 1.0 - strokes * tonalCoverage * Post.NprStyle.y;
    float edge = Post.NprStyle.x > 0.0 ? xdog(pixel, extent) : 1.0;
    float intervals = Post.NprTone.x - 1.0;
    vec3 quantized = floor(display * intervals + 0.5) / intervals;
    vec3 paper = mix(vec3(1), quantized, Post.NprStyle.z);
    vec3 ink = paper * mix(1.0, edge, Post.NprStyle.x) * transmission.x * transmission.y * transmission.z;
    OutputColor = vec4(toLinear(mix(display, ink, Post.NprStyle.w)), rgba.a);
}
#endif
