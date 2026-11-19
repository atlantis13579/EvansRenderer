#version 450
// Native Vulkan multisample reconstruction. Average linear radiance; choose one
// actual foreground sample for depth, material and rigid motion attributes.
// The filtered-color broadcast preserves the original per-sample opaque depth.
// Vulkan Specification: Multisampling and Sample Shading.
// https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-sampleshading
#ifdef EVANS_VERTEX_STAGE
void main()
{
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
#else
layout(set = 0, binding = 0, std140) uniform ResolveConstants { uvec4 Counts; } Resolve;
layout(set = 0, binding = 1) uniform texture2DMS RadianceSamples;
layout(set = 0, binding = 2) uniform texture2DMS DepthSamples;
layout(set = 0, binding = 3) uniform texture2DMS DiffuseSamples;
layout(set = 0, binding = 4) uniform texture2DMS NormalSamples;
layout(set = 0, binding = 5) uniform texture2DMS SpecularSamples;
layout(set = 0, binding = 6) uniform texture2DMS AmbientSamples;
layout(set = 0, binding = 7) uniform texture2DMS EmissionSamples;
layout(set = 0, binding = 8) uniform texture2DMS VelocitySamples;
layout(set = 0, binding = 9) uniform texture2D FilteredRadiance;
layout(set = 0, binding = 10) uniform sampler NearestSampler;
#if defined(EVANS_RESOLVE_MATERIAL)
layout(location = 0) out vec4 Diffuse;
layout(location = 1) out vec4 Normal;
layout(location = 2) out vec4 Specular;
layout(location = 3) out vec4 Ambient;
layout(location = 4) out vec4 Emission;
#elif !defined(EVANS_RESOLVE_DEPTH)
layout(location = 0) out vec4 Color;
#endif

int ClosestSurface(ivec2 pixel, out float depth)
{
    int selected = 0;
    depth = texelFetch(sampler2DMS(DepthSamples, NearestSampler), pixel, selected).r;
    for (int index = 1; index < int(Resolve.Counts.x); ++index)
    {
        float candidate = texelFetch(sampler2DMS(DepthSamples, NearestSampler), pixel, index).r;
        if (candidate < depth) { depth = candidate; selected = index; }
    }
    return selected;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
#if defined(EVANS_RESOLVE_DEPTH) || defined(EVANS_RESOLVE_MATERIAL) || defined(EVANS_RESOLVE_VELOCITY)
    float nearestDepth;
    int selected = ClosestSurface(pixel, nearestDepth);
#if defined(EVANS_RESOLVE_MATERIAL)
    gl_FragDepth = nearestDepth;
    Diffuse = texelFetch(sampler2DMS(DiffuseSamples, NearestSampler), pixel, selected);
    Normal = texelFetch(sampler2DMS(NormalSamples, NearestSampler), pixel, selected);
    Specular = texelFetch(sampler2DMS(SpecularSamples, NearestSampler), pixel, selected);
    Ambient = texelFetch(sampler2DMS(AmbientSamples, NearestSampler), pixel, selected);
    Emission = texelFetch(sampler2DMS(EmissionSamples, NearestSampler), pixel, selected);
#elif defined(EVANS_RESOLVE_DEPTH)
    gl_FragDepth = nearestDepth;
#else
    Color = texelFetch(sampler2DMS(VelocitySamples, NearestSampler), pixel, selected);
#endif
#elif defined(EVANS_RESTORE_COLOR)
    // One filtered opaque layer approximates mixed-coverage edges. Depth is not
    // written here, so transparent fragments still test the original samples.
    Color = texelFetch(sampler2D(FilteredRadiance, NearestSampler), pixel, 0);
#else
    vec4 sum = vec4(0.0);
    for (int index = 0; index < int(Resolve.Counts.x); ++index)
        sum += texelFetch(sampler2DMS(RadianceSamples, NearestSampler), pixel, index);
    Color = sum / float(Resolve.Counts.x);
#endif
}
#endif
