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
// Native ES single-map shadow depth. Williams, "Casting curved shadows on curved
// surfaces" (1978), doi:10.1145/800248.807402. Light-space depth is independent
// of surface illumination; coverage alone decides which fragments write it.
precision highp float;
precision highp int;
#ifdef EVANS_VERTEX_STAGE
layout(location = 0) in vec3 InPosition;
layout(location = 2) in vec4 InColor;
layout(location = 3) in vec2 InTexCoord;
uniform mat4 World;
uniform mat4 ViewProjection;
out vec2 MaterialUV;
out float VertexAlpha;
void main()
{
    vec4 projected = ViewProjection * World * vec4(InPosition, 1.0);
    // CPU projections use [0,w] depth; ES clips against [-w,w].
    projected.z = projected.z * 2.0 - projected.w;
    gl_Position = projected;
    MaterialUV = vec2(InTexCoord.x, 1.0 - InTexCoord.y);
    VertexAlpha = InColor.a;
    CullWireFace();
}
#else
in vec2 MaterialUV;
in float VertexAlpha;
uniform highp sampler2D DiffuseMap;
uniform highp sampler2D OpacityMap;
uniform vec4 MaterialParameters;
uniform vec4 TextureFlags;
uniform vec4 RenderParameters;
uniform vec4 ShadowDepthSize;
const float FlagThreshold = 0.5;
const float NoiseScale = 52.9829189;
const vec2 NoiseDirection = vec2(0.06711056, 0.00583715);
void main()
{
    float alpha = VertexAlpha * MaterialParameters.y;
    if (TextureFlags.x > FlagThreshold) alpha *= texture(DiffuseMap, MaterialUV).a;
    if (TextureFlags.z > FlagThreshold) alpha *= texture(OpacityMap, MaterialUV).r;
    alpha = clamp(alpha, 0.0, 1.0);
    if (MaterialParameters.z > 0.0)
    {
        if (alpha < MaterialParameters.z) discard;
    }
    else if (RenderParameters.w > FlagThreshold)
    {
        // Align deterministic partial coverage with the other APIs and CSM.
        // Depth textures retain ES orientation; only the noise coordinate flips.
        vec2 pixel = floor(gl_FragCoord.xy);
        pixel.y = ShadowDepthSize.y - pixel.y - 1.0;
        float threshold = fract(NoiseScale * fract(dot(pixel, NoiseDirection)));
        if (alpha <= 0.0 || alpha < threshold) discard;
    }
}
#endif
