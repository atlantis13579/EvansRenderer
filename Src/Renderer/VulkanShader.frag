#version 450

layout(location = 0) in vec3 WorldNormal;
layout(location = 1) in vec4 VertexColor;
layout(location = 2) in vec4 LightPosition;

layout(set = 0, binding = 0, std140) uniform DrawConstants
{
    mat4 World;
    mat4 ViewProjection;
    mat4 LightViewProjection;
    vec4 PipelineColor;
    vec4 Parameters;
} Constants;
layout(set = 0, binding = 1) uniform sampler2D ShadowMap;

layout(location = 0) out vec4 OutColor;

float SampleShadow()
{
    if (Constants.Parameters.x < 0.5 || abs(LightPosition.w) < 1.0e-7)
    {
        return 1.0;
    }

    vec3 projected = LightPosition.xyz / LightPosition.w;
    vec2 uv = vec2(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z < 0.0 || projected.z > 1.0)
    {
        return 1.0;
    }

    ivec2 size = textureSize(ShadowMap, 0);
    ivec2 center = ivec2(uv * vec2(size - ivec2(1)));
    float occluded = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            ivec2 pixel = clamp(center + ivec2(x, y), ivec2(0), size - ivec2(1));
            float storedDepth = texelFetch(ShadowMap, pixel, 0).r;
            occluded += projected.z > storedDepth + Constants.Parameters.w ? 1.0 : 0.0;
        }
    }
    return 1.0 - 0.55 * occluded / 9.0;
}

void main()
{
    if (Constants.Parameters.z > 0.5)
    {
        OutColor = Constants.PipelineColor;
        return;
    }

    vec3 light = normalize(vec3(0.45, 1.0, 0.35));
    float diffuse = abs(dot(normalize(WorldNormal), light));
    float lighting = 0.28 + 0.72 * diffuse * SampleShadow();
    OutColor = VertexColor * Constants.PipelineColor;
    OutColor.rgb *= lighting;
}
