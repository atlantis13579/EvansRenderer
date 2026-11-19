#version 300 es
// Radial pull resampling, independent of geometry depth and camera history.
// The first radial polynomial term follows OpenCV's camera distortion model:
// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
void main()
{
    const float TriangleExtent = 3.0;
    vec2 point = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
        gl_VertexID == 1 ? vec2(TriangleExtent, -1.0) : vec2(-1.0, TriangleExtent);
    gl_Position = vec4(point, 0.0, 1.0);
}
#else
uniform sampler2D SceneColor;
uniform vec4 InverseExtentStrengths;
uniform vec4 DistortionStrengthRadiusWeights;
out vec4 OutColor;
void main()
{
    const vec2 OpticalCenter = vec2(0.5);
    // Both gl_FragCoord and the input attachment use the lower-left origin.
    vec2 uv = gl_FragCoord.xy * InverseExtentStrengths.xy;
    vec2 centered = (uv - OpticalCenter) * 2.0;
    float normalizedRadiusSquared = dot(centered * centered, DistortionStrengthRadiusWeights.yz);
    float scale = 1.0 + DistortionStrengthRadiusWeights.x * normalizedRadiusSquared;
    vec2 sampleUV = OpticalCenter + (uv - OpticalCenter) * scale;
    // SRGB8_ALPHA8 decoding precedes linear filtering; alpha is resampled too.
    // The source texture uses clamp-to-edge. The output sRGB target encodes RGB.
    OutColor = textureLod(SceneColor, sampleUV, 0.0);
}
#endif
