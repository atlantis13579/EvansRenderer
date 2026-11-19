#version 300 es
#ifdef EVANS_VERTEX_STAGE

// Native line primitives have no face classification. Reject all endpoints of a
// back-facing original triangle, retaining the native line coverage and clipping.
layout(location=15) in highp vec4 WireFacePlane;
uniform highp int WireFaceCull;
uniform highp vec4 WireFaceCamera;
void CullWireFace()
{
    if (WireFaceCull!=0 && dot(WireFacePlane,WireFaceCamera)<=0.0)
        gl_Position=vec4(0.0,0.0,2.0,1.0);
}
#endif
// Parallel-Split Shadow Maps (Zhang, Sun and Nyman, GPU Gems 3, 2007).
// The CPU fits and snaps each cascade; this independent ES depth producer keeps
// material coverage in that projection. Lighting selects/blends the resulting maps.
// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus
precision highp float;
precision highp int;

#ifdef EVANS_VERTEX_STAGE
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World;
uniform mat4 ViewProjection;
out float VertexOpacity;
out vec2 MaterialUV;
void main()
{
    vec4 clip = ViewProjection * World * vec4(InPosition, 1.0);
    // Fitted matrices use [0,1] depth. ES clip space requires [-w,+w];
    // the viewport transform then stores the original [0,1] value in D32.
    clip.z = clip.z * 2.0 - clip.w;
    gl_Position = clip;
    VertexOpacity = InColor.a;
    MaterialUV = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    CullWireFace();
}
#else
in float VertexOpacity;
in vec2 MaterialUV;
uniform highp sampler2D DiffuseMap;
uniform highp sampler2D OpacityMap;
uniform vec4 MaterialParameters; // shininess, opacity, cutoff, outline width
uniform vec4 TextureFlags; // diffuse, normal, opacity, shading model
uniform vec4 RenderParameters; // shadow sampling, draw mode, light, alpha blending
uniform vec4 CascadeDepthSize; // width, height, reserved
const float EnabledThreshold = 0.5;
const float CoverageNoiseScale = 52.9829189;
const vec2 CoverageNoiseFrequency = vec2(0.06711056, 0.00583715);
void main()
{
    float diffuseAlpha = TextureFlags.x > EnabledThreshold ? texture(DiffuseMap, MaterialUV).a : 1.0;
    float opacityMap = TextureFlags.z > EnabledThreshold ? texture(OpacityMap, MaterialUV).r : 1.0;
    float coverage = clamp(VertexOpacity * MaterialParameters.y * diffuseAlpha * opacityMap, 0.0, 1.0);
    if (MaterialParameters.z > 0.0)
    {
        if (coverage < MaterialParameters.z) discard;
    }
    else if (RenderParameters.w > EnabledThreshold)
    {
        if (coverage <= 0.0) discard;
        // Interleaved gradient noise uses top-left integer pixels on every API.
        // Only this coverage pattern flips Y; ES depth sampling stays bottom-left.
        vec2 pixel = floor(gl_FragCoord.xy);
        pixel.y = CascadeDepthSize.y - pixel.y - 1.0;
        float threshold = fract(CoverageNoiseScale * fract(dot(pixel, CoverageNoiseFrequency)));
        if (coverage < threshold) discard;
    }
}
#endif
