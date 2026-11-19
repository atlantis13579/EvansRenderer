#version 450
// Screen-space radial distortion through an analytic destination-to-source map.
// Radius uses pixel-space half-diagonal units, not stretched UV distance.
// Reference: OpenCV, Camera Calibration and 3D Reconstruction, radial model.
// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float FarCorner = 3.0;
    vec2 corner = gl_VertexIndex == 0 ? vec2(-1.0, -1.0) :
        gl_VertexIndex == 1 ? vec2(FarCorner, -1.0) : vec2(-1.0, FarCorner);
    gl_Position = vec4(corner, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform PostConstants
{
    vec4 InverseExtentStrengths;
    uvec4 GrainSeedReserved;
    vec4 DistortionStrengthRadiusWeights;
} Post;
layout(set = 0, binding = 2) uniform texture2D SceneColor;
layout(set = 0, binding = 5) uniform sampler ColorSampler;
layout(location = 0) out vec4 OutputColor;
void main()
{
    const float ImageCenter = 0.5;
    vec2 destination = gl_FragCoord.xy * Post.InverseExtentStrengths.xy;
    vec2 offset = destination - ImageCenter;
    vec2 centered = offset * 2.0;
    float radiusSquared = dot(centered * centered, Post.DistortionStrengthRadiusWeights.yz);
    vec2 source = ImageCenter + offset * (1.0 + Post.DistortionStrengthRadiusWeights.x * radiusSquared);
    // SRGB image views and the linear clamp sampler interpolate linear RGB
    // and alpha, then the sRGB attachment encodes RGB. No depth write occurs.
    OutputColor = textureLod(sampler2D(SceneColor, ColorSampler), source, 0.0);
}
#endif
