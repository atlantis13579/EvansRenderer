#version 300 es
#ifdef EVANS_VERTEX_STAGE

// Native line primitives have no face classification. Reject all endpoints of a
// back-facing original triangle, retaining the native line coverage and clipping.
invariant gl_Position;
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp vec4 WireFaceCamera;
void CullWireFace()
{
    if (WireFaceCull!=0 && dot(WireFacePlane,WireFaceCamera)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}
#endif
// Rigid per-pixel motion vectors from current/previous homogeneous positions.
// Interpolate the previous clip position before dividing by W, replay opaque
// alpha coverage against read-only depth, and output top-left pixel displacement,
// reversed previous hardware depth and validity. This producer applies no shutter blur.
// Rosado, GPU Gems 3, "Motion Blur as a Post-Processing Effect" (2007), sec. 27.4.
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect
precision highp float;
precision highp int;
precision highp sampler2D;

uniform vec4 ExtentExposure; // full width/height, shutter/(2*interval), maximum radius
uniform uvec4 Options; // fullscreen stage, diffuse map, opacity map, valid history
const float PixelCenter = 0.5;
const float MinimumClipW = 1.0e-7;
const float FarDepth = 1.0;

uniform mat4 ObjectMatrix;
uniform mat4 CameraMatrix;
uniform mat4 PreviousClipMatrix;
uniform vec4 AlphaParameters; // combined material opacity, alpha cutoff
#ifdef EVANS_VERTEX_STAGE
// Match the invariant opaque position when replaying against its depth.

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec4 Color;
layout(location = 3) in vec2 UV;
out vec4 PreviousClip;
out vec2 SurfaceUV;
out float SurfaceAlpha;
void main()
{
    // Preserve the opaque vertex shader's operation order for depth replay.
    vec4 world = ObjectMatrix * vec4(Position, 1.0);
    vec4 clip = CameraMatrix * world;
    clip.z = clip.z * 2.0 - clip.w;
    gl_Position = clip;
    PreviousClip = PreviousClipMatrix * vec4(Position, 1.0);
    SurfaceUV = vec2(UV.x, 1.0 - UV.y);
    SurfaceAlpha = Color.a * AlphaParameters.x;
    CullWireFace();
}
#else
in vec4 PreviousClip;
in vec2 SurfaceUV;
in float SurfaceAlpha;
uniform sampler2D DiffuseAlpha;
uniform sampler2D OpacityMap;
layout(location = 0) out vec4 VelocityOutput;
const float MaximumHalfFloat = 65504.0;
void main()
{
    float alpha = SurfaceAlpha;
    if (Options.y != 0u) alpha *= texture(DiffuseAlpha, SurfaceUV).a;
    if (Options.z != 0u) alpha *= texture(OpacityMap, SurfaceUV).r;
    if (AlphaParameters.y > 0.0 && clamp(alpha, 0.0, 1.0) < AlphaParameters.y) discard;
    VelocityOutput = vec4(0.0);
    if (Options.w == 0u || PreviousClip.w <= MinimumClipW) return;
    // PreviousClip retains the renderer's zero-to-one depth convention. Only
    // gl_Position was converted to ES clip depth. Divide after interpolation.
    vec3 previousNdc = PreviousClip.xyz / PreviousClip.w;
    if (any(isnan(previousNdc)) || any(isinf(previousNdc)) || previousNdc.z < 0.0 || previousNdc.z > FarDepth) return;
    vec2 previousPixel = (previousNdc.xy * vec2(0.5, -0.5) + 0.5) * ExtentExposure.xy;
    vec2 currentPixel = vec2(gl_FragCoord.x, ExtentExposure.y - gl_FragCoord.y);
    // Jitter is in canonical top-left pixels, independent of the ES image origin.
    vec2 displacement = currentPixel - previousPixel - AlphaParameters.zw;
    if (any(greaterThan(abs(displacement), vec2(MaximumHalfFloat)))) return;
    // Store one minus hardware depth before half-float attachment conversion.
    VelocityOutput = vec4(displacement, 1.0-previousNdc.z, 1.0);
}
#endif
