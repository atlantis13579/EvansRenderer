#version 310 es
// OpenGL ES 3.1 multisampling. Resolve radiance in linear space; texture fetches
// decode sRGB samples and the destination attachment encodes the averaged result.
// Depth, material and velocity select the same nearest sample. This preserves
// one actual surface instead of inventing a depth/normal between silhouettes.
// OpenGL ES 3.1 specification: multisample textures and fragment operations.
// https://registry.khronos.org/OpenGL/specs/es/3.1/es_spec_3.1.pdf
precision highp float;
precision highp int;

#if defined(EVANS_VERTEX_STAGE)
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
uniform int SampleCount;
uniform highp sampler2DMS ColorSamples;
uniform highp sampler2DMS DepthSamples;
uniform highp sampler2DMS DiffuseSamples;
uniform highp sampler2DMS NormalSamples;
uniform highp sampler2DMS SpecularSamples;
uniform highp sampler2DMS AmbientSamples;
uniform highp sampler2DMS EmissionSamples;
uniform highp sampler2DMS VelocitySamples;
uniform highp sampler2D FilteredColor;

#if defined(EVANS_RESOLVE_MATERIAL)
layout(location = 0) out vec4 ResolvedDiffuse;
layout(location = 1) out vec4 ResolvedNormal;
layout(location = 2) out vec4 ResolvedSpecular;
layout(location = 3) out vec4 ResolvedAmbient;
layout(location = 4) out vec4 ResolvedEmission;
#elif !defined(EVANS_RESOLVE_DEPTH)
layout(location = 0) out vec4 ResolvedColor;
#endif

int ForemostSample(ivec2 pixel, out float closestDepth)
{
    int selected = 0;
    closestDepth = texelFetch(DepthSamples, pixel, selected).r;
    for (int sampleIndex = 1; sampleIndex < SampleCount; ++sampleIndex)
    {
        float candidate = texelFetch(DepthSamples, pixel, sampleIndex).r;
        if (candidate < closestDepth)
        {
            closestDepth = candidate;
            selected = sampleIndex;
        }
    }
    return selected;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#if defined(EVANS_RESOLVE_DEPTH) || defined(EVANS_RESOLVE_MATERIAL) || defined(EVANS_RESOLVE_VELOCITY)
    float depth;
    int selected = ForemostSample(pixel, depth);
#if defined(EVANS_RESOLVE_MATERIAL)
    gl_FragDepth = depth;
    ResolvedDiffuse = texelFetch(DiffuseSamples, pixel, selected);
    ResolvedNormal = texelFetch(NormalSamples, pixel, selected);
    ResolvedSpecular = texelFetch(SpecularSamples, pixel, selected);
    ResolvedAmbient = texelFetch(AmbientSamples, pixel, selected);
    ResolvedEmission = texelFetch(EmissionSamples, pixel, selected);
#elif defined(EVANS_RESOLVE_DEPTH)
    gl_FragDepth = depth;
#else
    ResolvedColor = texelFetch(VelocitySamples, pixel, selected);
#endif
#elif defined(EVANS_RESTORE_COLOR)
    // Broadcast filtered opaque radiance to all color samples. The attached
    // original opaque depth stays read-only, retaining transparent occlusion.
    // A single filtered background color approximates mixed-coverage edges.
    ResolvedColor = texelFetch(FilteredColor, pixel, 0);
#else
    vec4 radiance = vec4(0.0);
    for (int sampleIndex = 0; sampleIndex < SampleCount; ++sampleIndex)
        radiance += texelFetch(ColorSamples, pixel, sampleIndex);
    ResolvedColor = radiance / float(SampleCount);
#endif
}
#endif
