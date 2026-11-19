#version 450
// Rigid per-pixel motion vectors from current/previous homogeneous positions.
// Interpolate the previous clip position before dividing by W, replay opaque
// alpha coverage against read-only depth, and output top-left pixel displacement,
// reversed previous hardware depth and validity. This producer applies no shutter blur.
// Rosado, GPU Gems 3, "Motion Blur as a Post-Processing Effect" (2007), sec. 27.4.
// https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect
layout(set = 0, binding = 0, std140, row_major) uniform MotionConstants
{
    mat4 World;
    mat4 ViewProjection;
    mat4 PreviousWorldViewProjection;
    mat4 InverseViewProjection;
    vec4 DepthPlane;
    vec4 ViewportExposure; // width, height, shutter / (2 * interval), radius limit
    vec4 Material; // object/material opacity, alpha cutoff
    uvec4 Options; // filter pass, diffuse texture, opacity texture, history valid
} Motion;

const float HalfPixel = 0.5;
const float MinimumHomogeneousW = 1.0e-7;
const float ClearDepth = 1.0;

#ifdef EVANS_VERTEX_STAGE
// Match the invariant opaque position when replaying against its depth.
invariant gl_Position;
precise gl_Position;
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
layout(location = 0) out vec4 PreviousClip;
layout(location = 1) out vec2 TexCoord;
layout(location = 2) out float Opacity;
void main()
{
    // Match opaque vertex arithmetic so depth replay does not acquire holes.
    vec4 position = Motion.World * vec4(InPosition, 1.0);
    gl_Position = Motion.ViewProjection * position;
    gl_Position.y = -gl_Position.y;
    PreviousClip = Motion.PreviousWorldViewProjection * vec4(InPosition, 1.0);
    TexCoord = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    Opacity = InColor.a * Motion.Material.x;
}
#else
layout(location = 0) in vec4 PreviousClip;
layout(location = 1) in vec2 TexCoord;
layout(location = 2) in float Opacity;
layout(set = 0, binding = 2) uniform texture2D DiffuseMap;
layout(set = 0, binding = 4) uniform texture2D OpacityMap;
layout(set = 0, binding = 5) uniform sampler MaterialSampler;
layout(location = 0) out vec4 OutVelocity;
const float MaximumHalfFloat = 65504.0;
void main()
{
    float alpha = Opacity;
    if (Motion.Options.y != 0u) alpha *= texture(sampler2D(DiffuseMap, MaterialSampler), TexCoord).a;
    if (Motion.Options.z != 0u) alpha *= texture(sampler2D(OpacityMap, MaterialSampler), TexCoord).r;
    if (Motion.Material.y > 0.0 && clamp(alpha, 0.0, 1.0) < Motion.Material.y) discard;
    OutVelocity = vec4(0.0);
    if (Motion.Options.w == 0u || PreviousClip.w <= MinimumHomogeneousW) return;
    // Interpolate clip positions, not pre-divided NDC: the latter distorts
    // correspondence on triangles with nonuniform perspective depth.
    vec3 previousNdc = PreviousClip.xyz / PreviousClip.w;
    if (any(isnan(previousNdc)) || any(isinf(previousNdc)) ||
        previousNdc.z < 0.0 || previousNdc.z > ClearDepth) return;
    vec2 previousPixel = (previousNdc.xy * vec2(HalfPixel, -HalfPixel) + HalfPixel) * Motion.ViewportExposure.xy;
    // Subtract sampling jitter after converting to the common pixel convention.
    vec2 velocity = gl_FragCoord.xy - previousPixel - Motion.Material.zw;
    if (any(greaterThan(abs(velocity), vec2(MaximumHalfFloat)))) return;
    // Reversed device depth preserves relative precision in the half-float target.
    OutVelocity = vec4(velocity, 1.0-previousNdc.z, 1.0);
}
#endif
